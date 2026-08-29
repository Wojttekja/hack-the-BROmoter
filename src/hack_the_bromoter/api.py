"""
api.py -- klient HTTP do API hackathonu "Hack the Promoter".

Jedna funkcja na endpoint, bez zaleznosci poza biblioteka standardowa.
Klucz API czytany jest z pliku `.env` (zmienna `HYPPE_API_KEY`) albo ze
srodowiska. Zobacz `.env.example`.

Uzycie:

    from hack_the_bromoter import api

    api.me()
    dziki = api.dziki()["sekwencja"]
    api.sedzia(dziki, kandydat, nazwa_b="kandydat")
    api.wgraj(api.zbuduj_fasta([("wariant_1", seq), ...]))
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BAZOWY_URL = "https://hyppe.futura.foundation"

# urllib bez tego naglowka dostaje od Cloudflare "error code: 1010".
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

ZASADY = frozenset("ACGTN")
DLUGOSC_PROMOTORA = 800
MAX_UDZIAL_N = 0.10
LIMIT_OCENIANYCH = 100


class BladAPI(RuntimeError):
    """Odpowiedz inna niz 200 albo blad sieci."""

    def __init__(self, kod: int, tresc: Any, sciezka: str) -> None:
        super().__init__(f"{sciezka}: HTTP {kod}: {tresc}")
        self.kod = kod
        self.tresc = tresc
        self.sciezka = sciezka


# --------------------------------------------------------------------------
# konfiguracja
# --------------------------------------------------------------------------


def wczytaj_env(sciezka: str | Path | None = None) -> dict[str, str]:
    """Minimalny parser `.env` (KLUCZ=wartosc, `#` to komentarz)."""
    if sciezka is None:
        korzen = Path(__file__).resolve().parents[2]
        sciezka = korzen / ".env"
    sciezka = Path(sciezka)
    if not sciezka.is_file():
        return {}
    pary: dict[str, str] = {}
    for linia in sciezka.read_text(encoding="utf-8").splitlines():
        linia = linia.strip()
        if not linia or linia.startswith("#") or "=" not in linia:
            continue
        nazwa, _, wartosc = linia.partition("=")
        wartosc = wartosc.strip()
        if len(wartosc) >= 2 and wartosc[0] == wartosc[-1] and wartosc[0] in "\"'":
            wartosc = wartosc[1:-1]
        pary[nazwa.strip()] = wartosc
    return pary


def klucz_api(klucz: str | None = None) -> str:
    """Klucz z argumentu, ze srodowiska albo z `.env`."""
    if klucz:
        return klucz
    z_env = os.environ.get("HYPPE_API_KEY")
    if z_env:
        return z_env
    z_pliku = wczytaj_env().get("HYPPE_API_KEY", "")
    if not z_pliku or z_pliku == "YOUR_KEY":
        raise RuntimeError(
            "Brak klucza API. Wpisz HYPPE_API_KEY=... w pliku .env "
            "albo ustaw zmienna srodowiskowa."
        )
    return z_pliku


def bazowy_url() -> str:
    return (
        os.environ.get("HYPPE_API_URL")
        or wczytaj_env().get("HYPPE_API_URL")
        or BAZOWY_URL
    ).rstrip("/")


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


def wolaj(
    sciezka: str,
    dane: dict[str, Any] | None = None,
    *,
    klucz: str | None = None,
    limit: float = 900,
    prob: int = 6,
) -> tuple[int, Any]:
    """(kod_http, odpowiedz). `dane` != None -> POST z JSON-em.

    Ponawia 503 (kolejka do GPU) oraz 429 poza `/wgraj`, z backoffem.
    """
    url = bazowy_url() + sciezka
    naglowek = klucz_api(klucz)
    for nr in range(prob):
        z = urllib.request.Request(url)
        z.add_header("X-API-Key", naglowek)
        z.add_header("User-Agent", UA)
        if dane is not None:
            z.add_header("Content-Type", "application/json")
            z.data = json.dumps(dane).encode()
        try:
            with urllib.request.urlopen(z, timeout=limit) as o:
                return o.status, json.loads(o.read().decode())
        except urllib.error.HTTPError as e:
            tresc = e.read().decode("utf-8", "replace")
            try:
                tresc = json.loads(tresc)
            except json.JSONDecodeError:
                pass
            ponawialne = e.code == 503 or (e.code == 429 and sciezka != "/wgraj")
            if not ponawialne or nr == prob - 1:
                return e.code, tresc
            czekaj = float(e.headers.get("Retry-After") or 0) or 0.4 * 2**nr
            time.sleep(min(czekaj, 8.0))
        except urllib.error.URLError as e:
            if nr == prob - 1:
                return 0, f"siec: {e}"
            time.sleep(0.4 * 2**nr)
    return 0, "wyczerpano proby"  # nieosiagalne


def _wolaj_ok(sciezka: str, dane: dict[str, Any] | None = None, **kw: Any) -> Any:
    kod, odp = wolaj(sciezka, dane, **kw)
    if kod != 200:
        raise BladAPI(kod, odp, sciezka)
    return odp


# --------------------------------------------------------------------------
# endpointy
# --------------------------------------------------------------------------


def me(*, klucz: str | None = None) -> dict[str, Any]:
    """`GET /me` -- stan klucza, limity, zuzycie dzienne. Bez limitu."""
    return _wolaj_ok("/me", klucz=klucz)


def dziki(*, klucz: str | None = None) -> dict[str, Any]:
    """`GET /dziki` -- naturalny promotor `pks1`, 800 pz. Bez limitu.

    Zwraca m.in. `sekwencja`, `nazwa`, `gen`, `genom`, `dlugosc`, `sha256_12`.
    """
    return _wolaj_ok("/dziki", klucz=klucz)


def dzika_sekwencja(*, klucz: str | None = None) -> str:
    """Sama sekwencja z `GET /dziki`."""
    return dziki(klucz=klucz)["sekwencja"]


def sedzia(
    a: str,
    b: str,
    *,
    nazwa_a: str = "a",
    nazwa_b: str = "b",
    klucz: str | None = None,
) -> dict[str, Any]:
    """`POST /sedzia` -- ktora z pary sekwencji jest silniejsza. 600/min.

    Zwraca `silniejsza` (nazwa) oraz `silniejsza_idx` (0 albo 1).
    """
    return _wolaj_ok(
        "/sedzia",
        {"a": a, "b": b, "nazwa_a": nazwa_a, "nazwa_b": nazwa_b},
        klucz=klucz,
    )


def czy_b_silniejsza(a: str, b: str, **kw: Any) -> bool:
    """True, gdy Sedzia wskazal `b`. Remis liczy sie jako False."""
    return sedzia(a, b, **kw).get("silniejsza_idx") == 1


def nawigator_mapa(
    sekwencja: str,
    *,
    od: int = 0,
    ile: int = DLUGOSC_PROMOTORA,
    klucz: str | None = None,
) -> dict[str, Any]:
    """`POST /nawigator/mapa` -- opis kazdej pozycji sekwencji. 600/min.

    Kluczowe pola odpowiedzi: `pozycje` (rekon, warstwy, zmien_na, wagaP),
    `kompakt` (te same dane jako rownolegle tablice), `gatunek`,
    `rekon_frakcja`, `zmian_pod_gatunek`, `rozklad_warstw`.
    """
    return _wolaj_ok(
        "/nawigator/mapa",
        {"sekwencja": sekwencja, "od": od, "ile": ile},
        klucz=klucz,
    )


def nawigator_edycje(
    sekwencja: str,
    *,
    poziom: int = 2,
    ile_kodow: int = 8,
    opcji: int = 8,
    ziarno: int | None = None,
    klucz: str | None = None,
) -> dict[str, Any]:
    """`POST /nawigator/edycje` -- warianty przez zmiane kodow latentu. 600/min.

    poziom 0 = L1 (50 slotow / 16 bp / alfabet 4),
    poziom 1 = L2 (200 / 4 / 8),
    poziom 2 = L3 (400 / 2 / 4).
    Zwraca `opcje` (kazda z `nr`, `sekwencja`, `zmiany`) i metadane warstwy.
    """
    dane: dict[str, Any] = {
        "sekwencja": sekwencja,
        "poziom": poziom,
        "ile_kodow": ile_kodow,
        "opcji": opcji,
    }
    if ziarno is not None:
        dane["ziarno"] = ziarno
    return _wolaj_ok("/nawigator/edycje", dane, klucz=klucz)


def wgraj(fasta: str, *, klucz: str | None = None) -> dict[str, Any]:
    """`POST /wgraj` -- zgloszenie pliku FASTA do oceny. Raz na 5 minut.

    Zwraca `filtrowanie` (co odrzucono i dlaczego, z identyfikatorami),
    `ocenionych`, `pozycja_top10`, `pozycja_top100`, `punkty_razem`.
    Nie ponawia 429 -- to odstep miedzy zgloszeniami, nie przeciazenie.
    """
    return _wolaj_ok("/wgraj", {"fasta": fasta}, klucz=klucz)


def ranking(*, klucz: str | None = None) -> dict[str, Any]:
    """`GET /ranking` -- tablica wynikow, tylko punkty. Bez limitu."""
    return _wolaj_ok("/ranking", klucz=klucz)


# --------------------------------------------------------------------------
# pomocnicze: FASTA i walidacja pod filtry serwera
# --------------------------------------------------------------------------


def waliduj_sekwencje(sekwencja: str) -> list[str]:
    """Lista powodow, dla ktorych serwer pominalby sekwencje (pusta = ok)."""
    powody = []
    if len(sekwencja) != DLUGOSC_PROMOTORA:
        powody.append(f"dlugosc {len(sekwencja)} != {DLUGOSC_PROMOTORA}")
    obce = sorted(set(sekwencja.upper()) - ZASADY)
    if obce:
        powody.append("znaki poza ACGTN: " + "".join(obce))
    if sekwencja:
        udzial_n = sekwencja.upper().count("N") / len(sekwencja)
        if udzial_n > MAX_UDZIAL_N:
            powody.append(f"N = {udzial_n:.1%} > {MAX_UDZIAL_N:.0%}")
    return powody


def zbuduj_fasta(
    sekwencje: list[tuple[str, str]],
    *,
    limit: int = LIMIT_OCENIANYCH,
    pomin_niepoprawne: bool = True,
) -> str:
    """Buduje tekst FASTA z par (nazwa, sekwencja).

    Odsiewa duplikaty i -- domyslnie -- sekwencje, ktore i tak odpadlyby na
    filtrach serwera. Przycina do `limit`, bo oceniane jest pierwsze 100.
    """
    linie: list[str] = []
    widziane: set[str] = set()
    for nazwa, seq in sekwencje:
        if len(linie) // 2 >= limit:
            break
        seq = seq.strip().upper()
        if seq in widziane:
            continue
        if pomin_niepoprawne and waliduj_sekwencje(seq):
            continue
        widziane.add(seq)
        linie += [">" + nazwa, seq]
    return "\n".join(linie)


def czytaj_fasta(tekst: str) -> list[tuple[str, str]]:
    """Parsuje tekst FASTA na liste par (nazwa, sekwencja)."""
    rekordy: list[tuple[str, str]] = []
    nazwa: str | None = None
    kawalki: list[str] = []
    for linia in tekst.splitlines():
        linia = linia.strip()
        if not linia:
            continue
        if linia.startswith(">"):
            if nazwa is not None:
                rekordy.append((nazwa, "".join(kawalki)))
            nazwa, kawalki = linia[1:].strip(), []
        elif nazwa is not None:
            kawalki.append(linia)
    if nazwa is not None:
        rekordy.append((nazwa, "".join(kawalki)))
    return rekordy


def zastosuj_rekomendacje(sekwencja: str, mapa: dict[str, Any]) -> str:
    """Nanosi na sekwencje wszystkie `zmien_na` z odpowiedzi `/nawigator/mapa`.

    Pozycje w odpowiedzi sa liczone od 1.
    """
    zasady = list(sekwencja)
    for w in mapa["pozycje"]:
        if w["zmien_na"] != ".":
            zasady[w["poz"] - 1] = w["zmien_na"]
    return "".join(zasady)


def wypisz_ranking(tabela: dict[str, Any] | None = None) -> None:
    """Wypisuje tablice wynikow w formie tabelki."""
    t = tabela if tabela is not None else ranking()
    print(
        f"druzyn {t['n_druzyn']} | startujacych {t['n_startujacych']} "
        f"| twoja pozycja: {t['twoja_pozycja']}"
    )
    print()
    naglowki = ("poz", "druzyna", "ocen", "TOP10", "TOP100", "razem", "wgranie")
    print(
        f"{naglowki[0]:<4} {naglowki[1]:<13} {naglowki[2]:<6} {naglowki[3]:<8} "
        f"{naglowki[4]:<9} {naglowki[5]:<7} {naglowki[6]}"
    )
    print("-" * 74)
    for x in t["ranking"]:
        print(
            f"{x['pozycja']:<4} {x['druzyna']:<13} {x['ocenionych']:<6} "
            f"{x['punkty_top10']:<8} {x['punkty_top100']:<9} "
            f"{x['punkty_razem']:<7} {x['wgranie_o'] or '-'}"
        )
