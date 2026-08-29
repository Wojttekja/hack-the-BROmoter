"""
api.py -- thin Python client for the HYPPE "Hack the Promoter" API.

One function per endpoint, stdlib only (urllib), with automatic retries on
503 (GPU queue full) and 429 (rate limit, except for ``/wgraj``).

All ``HYPPE_API_KEY*`` entries of the environment / the ``.env`` file in the
project root are used (see ``.env.example``). Every key has its own per-minute
limit, so requests go round-robin over the pool and a key that answers 429 is
parked for the rest of its minute while the others keep working; the client
only sleeps when every key is cooling down.

Endpoint names, request fields and response keys are kept in the original
Polish, exactly as the server expects them.

Typical use:

    from hack_the_bromoter.api import me, dziki, sedzia, wgraj

    print(me()["druzyna"])
    wild = dziki()["sekwencja"]
    print(sedzia(wild, candidate)["silniejsza_idx"])
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "ApiError",
    "HyppeClient",
    "apply_recommendations",
    "build_fasta",
    "check_sequence",
    "dziki",
    "get_client",
    "is_b_stronger",
    "load_env",
    "load_keys",
    "me",
    "me_all",
    "nawigator_edycje",
    "nawigator_mapa",
    "parse_fasta",
    "print_ranking",
    "ranking",
    "sedzia",
    "wgraj",
    "wild_sequence",
]

DEFAULT_URL = "https://hyppe.futura.foundation"

# Every environment variable whose name starts with this is treated as a key.
KEY_ENV_PREFIX = "HYPPE_API_KEY"
PLACEHOLDER_KEYS = frozenset({"", "YOUR_KEY"})

# How long a key is parked after a 429 when the server sends no Retry-After;
# the documented limits are per minute.
KEY_COOLDOWN = 60.0
# A 429 from /wgraj means the 5 minute upload cooldown of that key.
UPLOAD_COOLDOWN = 300.0

# Transient gateway failures worth another attempt: 502/504 come from the
# proxy in front of the API when the GPU box is slow or overloaded, and 503 is
# the documented "queue full". None of them say anything about the request.
RETRYABLE_STATUS = frozenset({502, 503, 504})

# Cloudflare in front of the API rejects requests without a browser-like
# User-Agent with "error code: 1010".
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

SEQUENCE_LENGTH = 800
ALPHABET = frozenset("ACGTN")
MAX_N_FRACTION = 0.10
MAX_SCORED_SEQUENCES = 100
MAX_FASTA_CHARS = 2_000_000


class ApiError(RuntimeError):
    """Non-200 answer from the API (or a network failure)."""

    def __init__(self, status: int, path: str, body: Any) -> None:
        super().__init__(f"{path} -> HTTP {status}: {body!r}")
        self.status = status
        self.path = path
        self.body = body


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def load_env(path: str | Path | None = None) -> dict[str, str]:
    """Parse a ``.env`` file into a dict without overwriting real env vars.

    Values already present in ``os.environ`` win. Missing file -> empty dict.
    With no ``path``, walks up from this file looking for ``.env``.
    """
    if path is None:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / ".env"
            if candidate.is_file():
                path = candidate
                break
        else:
            return {}

    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                values[key] = value
    return values


def load_keys(path: str | Path | None = None) -> list[str]:
    """Collect every API key from the environment and from ``.env``.

    Any variable named ``HYPPE_API_KEY``, ``HYPPE_API_KEY_1``, ... counts;
    real environment variables win over the file, placeholders and duplicates
    are dropped. The bare name comes first, the numbered ones in order.
    """
    env = load_env(path)
    merged = dict(env)
    merged.update(os.environ)

    def order(name: str) -> tuple[int, str]:
        suffix = name[len(KEY_ENV_PREFIX) :].lstrip("_")
        return (int(suffix), "") if suffix.isdigit() else (10**6, name)

    keys: list[str] = []
    for name in sorted(
        (n for n in merged if n.startswith(KEY_ENV_PREFIX)), key=order
    ):
        value = merged[name].strip()
        if value not in PLACEHOLDER_KEYS and value not in keys:
            keys.append(value)
    return keys


class _KeyPool:
    """Round-robin over API keys, skipping the ones that are rate limited."""

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = list(keys)
        self._ready_at = [0.0] * len(self.keys)
        self._next = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.keys)

    def acquire(self) -> tuple[int, str]:
        """Return ``(index, key)`` of the next usable key.

        Blocks only while *all* keys are cooling down.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                for offset in range(len(self.keys)):
                    index = (self._next + offset) % len(self.keys)
                    if self._ready_at[index] <= now:
                        self._next = (index + 1) % len(self.keys)
                        return index, self.keys[index]
                wait = min(self._ready_at) - now
            time.sleep(min(max(wait, 0.05), 8.0))

    def park(self, index: int, seconds: float) -> None:
        """Take a key out of rotation for ``seconds``."""
        with self._lock:
            self._ready_at[index] = max(
                self._ready_at[index], time.monotonic() + seconds
            )

    def cooldowns(self) -> list[float]:
        """Seconds left before each key is usable again (0 = ready now)."""
        with self._lock:
            now = time.monotonic()
            return [max(0.0, ready - now) for ready in self._ready_at]


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
class HyppeClient:
    """HTTP client cycling over every configured API key."""

    def __init__(
        self,
        api_key: str | Iterable[str] | None = None,
        base_url: str | None = None,
        timeout: float = 900.0,
        attempts: int = 6,
    ) -> None:
        env = load_env()
        if api_key is None:
            keys = load_keys()
        elif isinstance(api_key, str):
            keys = [api_key]
        else:
            keys = [k for k in api_key if k]
        if not keys:
            raise ValueError(
                "No API key. Put HYPPE_API_KEY_1=... in .env "
                "(copy .env.example) or pass api_key=..."
            )
        self.keys = _KeyPool(keys)
        self.base_url = (
            base_url
            or os.environ.get("HYPPE_API_URL")
            or env.get("HYPPE_API_URL")
            or DEFAULT_URL
        ).rstrip("/")
        self.timeout = timeout
        self.attempts = attempts

    # -- low level ---------------------------------------------------------
    def request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        key_index: int | None = None,
    ) -> Any:
        """Call ``path``; a non-None ``payload`` makes it a JSON POST.

        Each attempt takes the next key off the pool. Returns the decoded
        JSON body, raises :class:`ApiError` otherwise. Retries
        `RETRYABLE_STATUS` always and 429 everywhere except ``/wgraj`` (where the cooldown is 5 minutes and
        waiting it out inline makes no sense) -- a 429 parks that one key and
        the retry immediately goes out on another.

        ``key_index`` pins the call to one key instead of using the rotation.
        """
        last: tuple[int, Any] = (0, "no attempt made")
        for attempt in range(self.attempts):
            if key_index is None:
                index, api_key = self.keys.acquire()
            else:
                index, api_key = key_index, self.keys.keys[key_index]
            request = urllib.request.Request(self.base_url + path)
            request.add_header("X-API-Key", api_key)
            request.add_header("User-Agent", USER_AGENT)
            if payload is not None:
                request.add_header("Content-Type", "application/json")
                request.data = json.dumps(payload).encode()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                    return json.loads(answer.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                try:
                    body = json.loads(body)
                except ValueError:
                    pass
                last = (exc.code, body)
                retry_after = float(exc.headers.get("Retry-After") or 0)
                if exc.code == 429:
                    default = UPLOAD_COOLDOWN if path == "/wgraj" else KEY_COOLDOWN
                    self.keys.park(index, retry_after or default)
                retryable = (exc.code in RETRYABLE_STATUS
                             or (exc.code == 429 and path != "/wgraj"))
                if not retryable or attempt == self.attempts - 1:
                    raise ApiError(exc.code, path, body) from exc
                if exc.code == 429:
                    # The key is parked; the next attempt picks another one and
                    # acquire() waits only if the whole pool is cooling down.
                    continue
                time.sleep(min(retry_after or 0.4 * 2**attempt, 8.0))
            except (urllib.error.URLError, http.client.HTTPException) as exc:
                # http.client raises IncompleteRead when a response is cut
                # short -- seen on /nawigator/mapa, whose bodies run to ~85 kB
                # and get truncated under concurrency. It is a transport
                # failure like any other URLError, so it retries the same way;
                # it is *not* a subclass of URLError, so it needs naming here
                # or it escapes the client entirely.
                last = (0, f"network: {exc!r}")
                if attempt == self.attempts - 1:
                    raise ApiError(0, path, f"network: {exc!r}") from exc
                time.sleep(0.4 * 2**attempt)
        raise ApiError(last[0], path, last[1])

    # -- endpoints ---------------------------------------------------------
    def me(self) -> dict[str, Any]:
        """GET /me -- key state, per-minute limits, daily usage. No limit."""
        return self.request("/me")

    def me_all(self) -> list[dict[str, Any]]:
        """GET /me once per key -- the state of the whole pool. No limit."""
        return [self.request("/me", key_index=i) for i in range(len(self.keys))]

    def dziki(self) -> dict[str, Any]:
        """GET /dziki -- the wild-type ``pks1`` promoter, 800 bp. No limit.

        Keys: ``sekwencja``, ``nazwa``, ``gen``, ``genom``, ``dlugosc``,
        ``sha256_12``.
        """
        return self.request("/dziki")

    def sedzia(
        self,
        a: str,
        b: str,
        nazwa_a: str = "a",
        nazwa_b: str = "b",
    ) -> dict[str, Any]:
        """POST /sedzia -- which of the two sequences is stronger. 600/min.

        Returns ``silniejsza`` (the winner's name) and ``silniejsza_idx``
        (0 for ``a``, 1 for ``b``).
        """
        return self.request(
            "/sedzia",
            {"a": a, "b": b, "nazwa_a": nazwa_a, "nazwa_b": nazwa_b},
        )

    def nawigator_mapa(
        self,
        sekwencja: str,
        od: int = 0,
        ile: int = SEQUENCE_LENGTH,
    ) -> dict[str, Any]:
        """POST /nawigator/mapa -- per-position map of the sequence. 600/min.

        ``od``/``ile`` select the window. Each entry of ``pozycje`` carries
        ``poz`` (1-based), ``wej`` (input base), ``rekon`` (1 = the position
        is reproduced from the latent codes alone), ``warstwy`` (which of
        L1/L2/L3 move it), ``zmien_na`` (base suggested for the strain, ``.``
        = leave it) and ``wagaP`` (normalised gradient magnitude).
        """
        return self.request(
            "/nawigator/mapa",
            {"sekwencja": sekwencja, "od": od, "ile": ile},
        )

    def nawigator_edycje(
        self,
        sekwencja: str,
        poziom: int = 2,
        ile_kodow: int = 8,
        opcji: int = 8,
        ziarno: int | None = None,
    ) -> dict[str, Any]:
        """POST /nawigator/edycje -- variants made by editing latent codes. 600/min.

        ``poziom``: 0 = L1 (50 slots, 16 bp each, alphabet 4), 1 = L2 (200
        slots, 4 bp, alphabet 8), 2 = L3 (400 slots, 2 bp, alphabet 4).
        ``ile_kodow`` codes are changed and ``opcji`` variants come back in
        ``opcje``; ``ziarno`` makes the draw reproducible.
        """
        payload: dict[str, Any] = {
            "sekwencja": sekwencja,
            "poziom": poziom,
            "ile_kodow": ile_kodow,
            "opcji": opcji,
        }
        if ziarno is not None:
            payload["ziarno"] = ziarno
        return self.request("/nawigator/edycje", payload)

    def wgraj(self, fasta: str, key_index: int | None = None) -> dict[str, Any]:
        """POST /wgraj -- submit a FASTA file for scoring. Once per 5 minutes.

        ``fasta`` is the whole file as one string (JSON body, not multipart).
        The answer reports the filtering stats (``filtrowanie``) and the score
        (``ocenionych``, ``pozycja_top10``, ``pozycja_top100``,
        ``punkty_razem``). Only the best submission counts for the ranking.

        The cooldown is per key, so ``key_index`` pins the upload to one key
        (``zgloszenie_mozliwe_za_s`` in ``/me`` says when a key is free again)
        instead of spending whichever key the rotation hands out next.
        """
        if len(fasta) > MAX_FASTA_CHARS:
            raise ValueError(
                f"FASTA has {len(fasta)} characters, the limit is {MAX_FASTA_CHARS}"
            )
        return self.request("/wgraj", {"fasta": fasta}, key_index=key_index)

    def ranking(self) -> dict[str, Any]:
        """GET /ranking -- the scoreboard, points only. No limit."""
        return self.request("/ranking")


# --------------------------------------------------------------------------
# module-level convenience wrappers over one shared client
# --------------------------------------------------------------------------
_CLIENT: HyppeClient | None = None


def get_client(**kwargs: Any) -> HyppeClient:
    """Return the shared client, building it on first use.

    Any keyword argument (``api_key``, ``base_url``, ``timeout``,
    ``attempts``) forces a fresh client to be built and cached.
    """
    global _CLIENT
    if _CLIENT is None or kwargs:
        _CLIENT = HyppeClient(**kwargs)
    return _CLIENT


def me() -> dict[str, Any]:
    """GET /me -- see :meth:`HyppeClient.me`."""
    return get_client().me()


def me_all() -> list[dict[str, Any]]:
    """GET /me per key -- see :meth:`HyppeClient.me_all`."""
    return get_client().me_all()


def dziki() -> dict[str, Any]:
    """GET /dziki -- see :meth:`HyppeClient.dziki`."""
    return get_client().dziki()


def sedzia(a: str, b: str, nazwa_a: str = "a", nazwa_b: str = "b") -> dict[str, Any]:
    """POST /sedzia -- see :meth:`HyppeClient.sedzia`."""
    return get_client().sedzia(a, b, nazwa_a=nazwa_a, nazwa_b=nazwa_b)


def nawigator_mapa(
    sekwencja: str, od: int = 0, ile: int = SEQUENCE_LENGTH
) -> dict[str, Any]:
    """POST /nawigator/mapa -- see :meth:`HyppeClient.nawigator_mapa`."""
    return get_client().nawigator_mapa(sekwencja, od=od, ile=ile)


def nawigator_edycje(
    sekwencja: str,
    poziom: int = 2,
    ile_kodow: int = 8,
    opcji: int = 8,
    ziarno: int | None = None,
) -> dict[str, Any]:
    """POST /nawigator/edycje -- see :meth:`HyppeClient.nawigator_edycje`."""
    return get_client().nawigator_edycje(
        sekwencja, poziom=poziom, ile_kodow=ile_kodow, opcji=opcji, ziarno=ziarno
    )


def wgraj(fasta: str, key_index: int | None = None) -> dict[str, Any]:
    """POST /wgraj -- see :meth:`HyppeClient.wgraj`."""
    return get_client().wgraj(fasta, key_index=key_index)


def ranking() -> dict[str, Any]:
    """GET /ranking -- see :meth:`HyppeClient.ranking`."""
    return get_client().ranking()


# --------------------------------------------------------------------------
# helpers around the submission format
# --------------------------------------------------------------------------
def check_sequence(sequence: str) -> list[str]:
    """Return the reasons the server would drop this sequence (empty = fine)."""
    problems = []
    if len(sequence) != SEQUENCE_LENGTH:
        problems.append(f"length {len(sequence)} instead of {SEQUENCE_LENGTH} bp")
    bad = sorted(set(sequence.upper()) - ALPHABET)
    if bad:
        problems.append("characters outside ACGTN: " + ", ".join(bad))
    n_count = sequence.upper().count("N")
    if n_count > MAX_N_FRACTION * len(sequence):
        problems.append(
            f"{n_count} N in {len(sequence)} bp, above the "
            f"{MAX_N_FRACTION:.0%} threshold"
        )
    return problems


def build_fasta(
    named_sequences: dict[str, str] | list[tuple[str, str]],
    limit: int = MAX_SCORED_SEQUENCES,
    drop_invalid: bool = True,
) -> str:
    """Render ``(name, sequence)`` pairs as a FASTA string ready for /wgraj.

    Duplicate sequences and, with ``drop_invalid``, sequences the server would
    filter out are removed; the rest is truncated to ``limit`` records, since
    only the first 100 that pass the filters are ever scored.
    """
    items = (
        list(named_sequences.items())
        if isinstance(named_sequences, dict)
        else list(named_sequences)
    )

    lines: list[str] = []
    seen: set[str] = set()
    kept = 0
    for name, sequence in items:
        if kept >= limit:
            break
        sequence = sequence.upper()
        if sequence in seen:
            continue
        if drop_invalid and check_sequence(sequence):
            continue
        seen.add(sequence)
        lines += [">" + str(name).split()[0], sequence]
        kept += 1
    return "\n".join(lines) + "\n"


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into a list of ``(name, sequence)`` pairs."""
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name, chunks = line[1:].strip(), []
        elif name is not None:
            chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def apply_recommendations(sequence: str, mapa: dict[str, Any]) -> str:
    """Apply every ``zmien_na`` from a ``/nawigator/mapa`` answer to a sequence.

    Positions in the answer are 1-based.
    """
    bases = list(sequence)
    for entry in mapa["pozycje"]:
        if entry["zmien_na"] != ".":
            bases[entry["poz"] - 1] = entry["zmien_na"]
    return "".join(bases)


def is_b_stronger(a: str, b: str, **kwargs: Any) -> bool:
    """True when the judge picks ``b`` over ``a``."""
    return sedzia(a, b, **kwargs)["silniejsza_idx"] == 1


def wild_sequence() -> str:
    """Just the 800 bp wild-type ``pks1`` promoter string."""
    return dziki()["sekwencja"]


def print_ranking(table: dict[str, Any] | None = None) -> None:
    """Print the scoreboard as a table; fetches it when not given one."""
    t = table if table is not None else ranking()
    print(
        f"teams {t['n_druzyn']} | with a submission {t['n_startujacych']} "
        f"| your position: {t['twoja_pozycja']}"
    )
    print()
    print(
        f"{'pos':<4} {'team':<13} {'scored':<7} {'TOP10':<8} "
        f"{'TOP100':<9} {'total':<7} {'uploaded'}"
    )
    print("-" * 74)
    for row in t["ranking"]:
        print(
            f"{row['pozycja']:<4} {row['druzyna']:<13} {row['ocenionych']:<7} "
            f"{row['punkty_top10']:<8} {row['punkty_top100']:<9} "
            f"{row['punkty_razem']:<7} {row['wgranie_o'] or '-'}"
        )


if __name__ == "__main__":
    accounts = me_all()
    print("team      :", accounts[0]["druzyna"], "|", accounts[0]["uczestnik"])
    print("keys      :", len(accounts))
    for number, account in enumerate(accounts, 1):
        print(
            f"  key {number}  : limits/min {account['limity_na_minute']} "
            f"| upload in {account['zgloszenie_mozliwe_za_s']} s"
        )

    wild = dziki()
    print(
        f"wild type : {wild['nazwa']} | gene {wild['gen']} "
        f"| {wild['genom']} | {wild['dlugosc']} bp"
    )
