"""
make_promoters.py -- build data/promoters.fasta for the HYPPE hackathon.

Takes a genome FASTA + GFF3 (+ optionally a BLAST outfmt-6 table and a CDS
FASTA) and writes a seed promoter library: upstream regions with correct strand
handling, neighbour-gene clipping, data-driven "highly expressed" candidates
picked by codon bias, and negative controls.

Stdlib only. No bioconda, no pip, no network.

Typical run:

    python make_promoters.py \
        --genome  ncbi_dataset/data/GCA_000171015.2/GCA_000171015.2_*_genomic.fna \
        --gff     ncbi_dataset/data/GCA_000171015.2/genomic.gff \
        --blast   blast_hits.tsv \
        --cds     ncbi_dataset/data/GCA_000171015.2/cds_from_genomic.fna \
        --target  pks1 \
        --out     data/promoters.fasta

Outputs:
    <out>              FASTA of promoter sequences
    <out>.manifest.tsv one row per sequence: id, gene, class, coords, len, GC, ...
"""

from __future__ import annotations

import argparse
import gzip
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# genetic code (standard table 1), used for the codon-bias ranking
# --------------------------------------------------------------------------
BASES = "TCAG"
AAS = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
CODON2AA = {
    b1 + b2 + b3: AAS[i]
    for i, (b1, b2, b3) in enumerate(
        (x, y, z) for x in BASES for y in BASES for z in BASES
    )
}
# amino acids with a single codon carry no information about bias
_AA_COUNT = Counter(CODON2AA.values())
DEGENERATE = {c for c, a in CODON2AA.items() if a != "*" and _AA_COUNT[a] > 1}

COMPLEMENT = str.maketrans("ACGTNacgtnRYKMSWBDHVrykmswbdhv",
                           "TGCANtgcanYRMKSWVHDByrmkswvhdb")


# --------------------------------------------------------------------------
# small IO helpers
# --------------------------------------------------------------------------
def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def read_fasta(path):
    """Yield (header, sequence) pairs."""
    name, chunks = None, []
    with _open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].rstrip("\n"), []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def gc(seq: str) -> float:
    s = seq.upper()
    n = sum(s.count(b) for b in "ACGT")
    return (s.count("G") + s.count("C")) / n if n else 0.0


def parse_attrs(field: str) -> dict:
    """Parse GFF3 (k=v;) and GTF (k "v";) attribute columns."""
    out = {}
    for chunk in field.rstrip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            k, v = chunk.split("=", 1)
        elif " " in chunk:
            k, v = chunk.split(" ", 1)
        else:
            continue
        out[k.strip()] = v.strip().strip('"')
    return out


def first_of(d: dict, *keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None


# --------------------------------------------------------------------------
# GFF parsing
# --------------------------------------------------------------------------
class Gene:
    __slots__ = ("gid", "chrom", "start0", "end0", "strand", "name", "product")

    def __init__(self, gid, chrom, start0, end0, strand, name=None, product=None):
        self.gid = gid
        self.chrom = chrom
        self.start0 = start0       # 0-based, inclusive
        self.end0 = end0           # 0-based, exclusive
        self.strand = strand
        self.name = name
        self.product = product

    def __repr__(self):
        return f"<Gene {self.gid} {self.chrom}:{self.start0}-{self.end0}{self.strand}>"


def parse_gff(path):
    """Return (genes_by_id, protein_to_gene, gene_by_name)."""
    genes: dict[str, Gene] = {}
    tx_parent: dict[str, str] = {}      # transcript id -> gene id
    protein_to_gene: dict[str, str] = {}
    pending_cds = []                    # (protein_id, parent_id, locus_tag)
    product_of = {}

    tx_types = {"mRNA", "transcript", "ncRNA", "tRNA", "rRNA", "pseudogenic_transcript"}

    with _open(path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _src, feat, start, end, _sc, strand, _ph, attr = parts[:9]
            a = parse_attrs(attr)

            if feat in ("gene", "pseudogene"):
                gid = first_of(a, "locus_tag", "ID", "gene_id", "Name")
                if not gid:
                    continue
                gid = gid.replace("gene-", "", 1) if gid.startswith("gene-") else gid
                genes[gid] = Gene(
                    gid, chrom, int(start) - 1, int(end), strand,
                    name=first_of(a, "gene", "Name", "gene_name"),
                    product=None,
                )
                # remember the raw ID too, so Parent= chains resolve
                raw = a.get("ID")
                if raw and raw != gid:
                    tx_parent[raw] = gid

            elif feat in tx_types:
                tid = a.get("ID")
                par = a.get("Parent", "").split(",")[0]
                if tid:
                    lt = a.get("locus_tag")
                    tx_parent[tid] = lt or tx_parent.get(par, par)

            elif feat == "CDS":
                pid = first_of(a, "protein_id", "ID")
                par = a.get("Parent", "").split(",")[0]
                lt = a.get("locus_tag")
                if pid:
                    pending_cds.append((pid, par, lt))
                    if "product" in a:
                        product_of.setdefault(pid, a["product"])

    # resolve CDS -> gene, following Parent chains up to 3 hops
    for pid, par, lt in pending_cds:
        gid = lt
        if gid is None:
            cur, hops = par, 0
            while cur and hops < 3:
                if cur in genes:
                    gid = cur
                    break
                cur = tx_parent.get(cur)
                hops += 1
            if gid is None and cur in genes:
                gid = cur
        if gid and gid in genes:
            pid_clean = pid.replace("cds-", "", 1) if pid.startswith("cds-") else pid
            protein_to_gene[pid_clean] = gid
            protein_to_gene[pid] = gid
            if genes[gid].product is None and pid in product_of:
                genes[gid].product = product_of[pid]

    by_name = {}
    for g in genes.values():
        if g.name:
            by_name.setdefault(g.name.lower(), g.gid)

    return genes, protein_to_gene, by_name


# --------------------------------------------------------------------------
# upstream extraction
# --------------------------------------------------------------------------
def build_neighbour_index(genes):
    idx = defaultdict(list)
    for g in genes.values():
        idx[g.chrom].append((g.start0, g.end0, g.gid))
    for c in idx:
        idx[c].sort()
    return idx


def extract_upstream(gene, contigs, nbr_idx, up=1500, clip_neighbours=True):
    """Return (seq, lo, hi, truncated_by) for the region upstream of the CDS start."""
    chrom = gene.chrom
    if chrom not in contigs:
        return None
    seq_full = contigs[chrom]
    L = len(seq_full)
    truncated = "none"

    if gene.strand == "+":
        hi = gene.start0
        lo = max(0, hi - up)
        if lo == 0 and hi < up:
            truncated = "contig"
        if clip_neighbours:
            for s0, e0, gid in nbr_idx[chrom]:
                if gid == gene.gid or s0 >= hi:
                    continue
                if lo < e0 <= hi:
                    lo, truncated = e0, "neighbour"
        seq = seq_full[lo:hi]
    else:
        lo = gene.end0
        hi = min(L, lo + up)
        if hi == L and (L - lo) < up:
            truncated = "contig"
        if clip_neighbours:
            for s0, e0, gid in nbr_idx[chrom]:
                if gid == gene.gid or e0 <= lo:
                    continue
                if lo <= s0 < hi:
                    hi, truncated = s0, "neighbour"
                    break
        seq = revcomp(seq_full[lo:hi])

    return seq.upper(), lo, hi, truncated


# --------------------------------------------------------------------------
# codon-bias ranking (iteratively refined CAI, Carbone-style)
# --------------------------------------------------------------------------
def codon_counts_from_cds(path):
    """Return {gene_key: Counter(codon)} using locus_tag / protein_id from the header."""
    out = {}
    for header, seq in read_fasta(path):
        keys = re.findall(r"\[(locus_tag|protein_id|gene)=([^\]]+)\]", header)
        keyd = {k: v for k, v in keys}
        key = keyd.get("locus_tag") or keyd.get("protein_id")
        if key is None:
            key = header.split()[0].split("|")[-1]
        s = seq.upper().replace("U", "T")
        if len(s) < 300 or len(s) % 3:
            continue
        cnt = Counter(
            s[i:i + 3] for i in range(0, len(s) - 3, 3)  # drop the stop codon
        )
        cnt = Counter({c: n for c, n in cnt.items() if c in DEGENERATE})
        if sum(cnt.values()) >= 80:
            out[key] = cnt
    return out


def cai_weights(counter_iterable):
    totals = Counter()
    for c in counter_iterable:
        totals.update(c)
    by_aa = defaultdict(dict)
    for codon, n in totals.items():
        by_aa[CODON2AA[codon]][codon] = n
    w = {}
    for aa, d in by_aa.items():
        mx = max(d.values()) or 1
        for codon, n in d.items():
            w[codon] = max(n / mx, 0.01)
    return w


def cai_score(counts, w):
    tot = 0.0
    n = 0
    for codon, k in counts.items():
        if codon in w:
            import math
            tot += k * math.log(w[codon])
            n += k
    import math
    return math.exp(tot / n) if n else 0.0


def rank_by_codon_bias(cds_counts, rounds=3, top_frac=0.05, min_ref=50):
    keys = list(cds_counts)
    w = cai_weights(cds_counts.values())
    ranked = keys
    for _ in range(rounds):
        scores = {k: cai_score(cds_counts[k], w) for k in keys}
        ranked = sorted(keys, key=lambda k: scores[k], reverse=True)
        n_ref = max(min_ref, int(len(ranked) * top_frac))
        w = cai_weights(cds_counts[k] for k in ranked[:n_ref])
    scores = {k: cai_score(cds_counts[k], w) for k in keys}
    ranked = sorted(keys, key=lambda k: scores[k], reverse=True)
    return ranked, scores


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
def dinuc_shuffle(seq, rng, max_tries=25):
    """Altschul-Erikson dinucleotide-preserving shuffle."""
    s = seq.upper()
    if len(s) < 4:
        return s
    for _ in range(max_tries):
        edges = defaultdict(list)
        for a, b in zip(s, s[1:]):
            edges[a].append(b)
        last = s[-1]
        # pick a random last-edge per vertex pointing towards `last`, keeping a tree
        last_edge = {}
        ok = True
        for v in edges:
            if v == last:
                continue
            cands = [i for i, b in enumerate(edges[v])]
            if not cands:
                ok = False
                break
            last_edge[v] = rng.choice(cands)
        if not ok:
            continue
        # connectivity check: following last_edge from every vertex must reach `last`
        good = True
        for v in edges:
            if v == last:
                continue
            cur, hops = v, 0
            while cur != last and hops <= len(edges) + 1:
                if cur not in last_edge:
                    good = False
                    break
                cur = edges[cur][last_edge[cur]]
                hops += 1
            if cur != last:
                good = False
            if not good:
                break
        if not good:
            continue
        # shuffle the non-last edges, append the reserved last edge
        pools = {}
        for v, lst in edges.items():
            if v == last:
                rest = lst[:]
                rng.shuffle(rest)
                pools[v] = rest
            else:
                i = last_edge[v]
                reserved = lst[i]
                rest = lst[:i] + lst[i + 1:]
                rng.shuffle(rest)
                pools[v] = rest + [reserved]
        out = [s[0]]
        cur = s[0]
        ptr = defaultdict(int)
        for _ in range(len(s) - 1):
            pool = pools.get(cur)
            if not pool or ptr[cur] >= len(pool):
                out = None
                break
            nxt = pool[ptr[cur]]
            ptr[cur] += 1
            out.append(nxt)
            cur = nxt
        if out and len(out) == len(s):
            return "".join(out)
    # fall back to a plain shuffle rather than failing the run
    lst = list(s)
    rng.shuffle(lst)
    return "".join(lst)


def random_intergenic(contigs, nbr_idx, length, rng, tries=200):
    chroms = [c for c in nbr_idx if len(contigs.get(c, "")) > length * 4]
    if not chroms:
        return None
    for _ in range(tries):
        c = rng.choice(chroms)
        L = len(contigs[c])
        lo = rng.randrange(0, L - length)
        hi = lo + length
        if any(s0 < hi and e0 > lo for s0, e0, _ in nbr_idx[c]):
            continue
        seq = contigs[c][lo:hi].upper()
        if seq.count("N") > 0.01 * length:
            continue
        return seq, c, lo, hi
    return None


# --------------------------------------------------------------------------
# BLAST
# --------------------------------------------------------------------------
def best_blast_hits(path, max_evalue=1e-20, min_pident=30.0):
    """outfmt 6 -> {query_name: subject_id} keeping the top bitscore per query."""
    best = {}
    with _open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 12:
                continue
            q, s = f[0], f[1]
            pident, evalue, bits = float(f[2]), float(f[10]), float(f[11])
            if evalue > max_evalue or pident < min_pident:
                continue
            if q not in best or bits > best[q][1]:
                best[q] = (s, bits)
    return {q: v[0] for q, v in best.items()}


def tidy_query_name(q):
    q = q.split("|")[-1] if "|" in q else q
    q = re.split(r"[_.\s]", q)[0]
    return re.sub(r"[^A-Za-z0-9]", "", q) or "query"


def strip_prefixes(x):
    for p in ("cds-", "gene-", "rna-", "lcl|"):
        if x.startswith(p):
            x = x[len(p):]
    return x


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", required=True, help="genomic FASTA (.fna[.gz])")
    ap.add_argument("--gff", required=True, help="GFF3 annotation")
    ap.add_argument("--blast", help="BLAST outfmt 6 table (queries = named refs)")
    ap.add_argument("--cds", help="CDS FASTA, enables codon-bias seed picking")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra locus tags / gene names to include, e.g. TRIATDRAFT_1234")
    ap.add_argument("--target", default="pks1",
                    help="gene name or locus tag to mark as class=target")
    ap.add_argument("--up", type=int, default=1500, help="bp upstream of CDS start")
    ap.add_argument("--min-len", type=int, default=250,
                    help="drop promoters shorter than this after clipping")
    ap.add_argument("--n-cai", type=int, default=40,
                    help="how many top codon-bias genes to add")
    ap.add_argument("--n-mid", type=int, default=10,
                    help="how many mid-ranked genes to add as moderate controls")
    ap.add_argument("--n-shuffle", type=int, default=6)
    ap.add_argument("--n-random", type=int, default=6)
    ap.add_argument("--no-clip", action="store_true",
                    help="do NOT clip at upstream neighbour genes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/promoters.fasta")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log = lambda *a: print(*a, file=sys.stderr)

    log("[1/6] reading genome ...")
    contigs = {h.split()[0]: s for h, s in read_fasta(args.genome)}
    log(f"      {len(contigs)} contigs, {sum(len(s) for s in contigs.values()):,} bp")

    log("[2/6] parsing GFF ...")
    genes, prot2gene, by_name = parse_gff(args.gff)
    if not genes:
        sys.exit("ERROR: no gene features found in the GFF. Check the file.")
    nbr_idx = build_neighbour_index(genes)
    missing = {g.chrom for g in genes.values()} - set(contigs)
    if missing:
        log(f"      WARNING: {len(missing)} GFF seqids absent from the FASTA, "
            f"e.g. {sorted(missing)[:3]} -- those genes will be skipped")
    log(f"      {len(genes)} genes, {len(prot2gene)} protein->gene links")

    # ---- selection -------------------------------------------------------
    selected = {}   # gid -> (label, cls)

    def take(gid, label, cls):
        if gid in genes and gid not in selected:
            selected[gid] = (label, cls)
            return True
        return False

    def resolve(token):
        """token may be a locus tag, gene name, or protein id."""
        t = strip_prefixes(token.strip())
        if t in genes:
            return t
        if t in prot2gene:
            return prot2gene[t]
        base = t.split(".")[0]
        if base in prot2gene:
            return prot2gene[base]
        if t.lower() in by_name:
            return by_name[t.lower()]
        return None

    if args.blast:
        log("[3/6] mapping BLAST hits to genes ...")
        hits = best_blast_hits(args.blast)
        hit_n = 0
        for q, subj in sorted(hits.items()):
            gid = resolve(subj)
            if gid is None:
                log(f"      ! could not map subject {subj!r} (query {q}) to a gene")
                continue
            if take(gid, tidy_query_name(q), "named"):
                hit_n += 1
        log(f"      {hit_n} genes from {len(hits)} queries")
    else:
        log("[3/6] no --blast given, skipping named genes")

    for token in args.extra:
        gid = resolve(token)
        if gid is None:
            log(f"      ! --extra {token!r} not found")
        else:
            take(gid, token, "named")

    cai_scores = {}
    if args.cds:
        log("[4/6] ranking genes by codon bias ...")
        counts = codon_counts_from_cds(args.cds)
        if not counts:
            log("      ! no usable CDS records; skipping")
        else:
            ranked_keys, sc = rank_by_codon_bias(counts)
            ranked_gids, seen = [], set()
            for k in ranked_keys:
                gid = resolve(k)
                if gid and gid not in seen:
                    seen.add(gid)
                    ranked_gids.append(gid)
                    cai_scores[gid] = sc[k]
            log(f"      scored {len(ranked_gids)} genes; "
                f"top CAI {sc[ranked_keys[0]]:.3f}, bottom {sc[ranked_keys[-1]]:.3f}")
            added = 0
            for gid in ranked_gids:
                if added >= args.n_cai:
                    break
                if take(gid, f"cai{added + 1:02d}", "high_cai"):
                    added += 1
            mid = len(ranked_gids) // 2
            added = 0
            for gid in ranked_gids[mid:]:
                if added >= args.n_mid:
                    break
                if take(gid, f"mid{added + 1:02d}", "moderate"):
                    added += 1
    else:
        log("[4/6] no --cds given, skipping codon-bias seeds")

    # mark the target -- accept a locus tag, gene name, protein id, or the
    # label derived from a BLAST query (e.g. --target pks1 for query pks1_Tatro...)
    tgt_gid = resolve(args.target)
    if tgt_gid is None:
        want = tidy_query_name(args.target).lower()
        for gid, (label, _cls) in selected.items():
            if label.lower() == want or tidy_query_name(label).lower() == want:
                tgt_gid = gid
                break
    if tgt_gid:
        label = selected.get(tgt_gid, (args.target, None))[0]
        selected[tgt_gid] = (args.target, "target")
        log(f"      target {args.target} -> {tgt_gid} (was labelled {label!r})")
    else:
        log(f"      ! target {args.target!r} not resolved -- pass its locus tag via "
            f"--target, or add it with --extra")

    if not selected:
        sys.exit("ERROR: nothing selected. Give --blast and/or --cds and/or --extra.")

    # ---- extraction ------------------------------------------------------
    log(f"[5/6] extracting {args.up} bp upstream for {len(selected)} genes ...")
    records = []
    dropped = 0
    for gid, (label, cls) in selected.items():
        g = genes[gid]
        res = extract_upstream(g, contigs, nbr_idx, up=args.up,
                               clip_neighbours=not args.no_clip)
        if res is None:
            dropped += 1
            continue
        seq, lo, hi, trunc = res
        if len(seq) < args.min_len or seq.count("N") > 0.05 * max(len(seq), 1):
            dropped += 1
            continue
        records.append(dict(
            id=f"{label}_{gid}", gene=label, locus=gid, cls=cls,
            chrom=g.chrom, strand=g.strand, lo=lo, hi=hi, trunc=trunc,
            cai=cai_scores.get(gid), product=g.product or "", seq=seq,
        ))
    log(f"      kept {len(records)}, dropped {dropped} (too short / N-rich / no contig)")

    # ---- controls --------------------------------------------------------
    log("[6/6] building controls ...")
    strong = [r for r in records if r["cls"] in ("high_cai", "named")]
    strong.sort(key=lambda r: -(r["cai"] or 0))
    for i, r in enumerate(strong[:args.n_shuffle]):
        records.append(dict(
            id=f"shuf{i + 1:02d}_of_{r['gene']}", gene=f"shuffled_{r['gene']}",
            locus="-", cls="control_shuffled", chrom="-", strand=".",
            lo=-1, hi=-1, trunc="none", cai=None, product="dinucleotide shuffle",
            seq=dinuc_shuffle(r["seq"], rng),
        ))
    lens = [len(r["seq"]) for r in records if r["cls"] != "control_shuffled"]
    med = sorted(lens)[len(lens) // 2] if lens else args.up
    for i in range(args.n_random):
        hit = random_intergenic(contigs, nbr_idx, med, rng)
        if hit is None:
            break
        seq, c, lo, hi = hit
        records.append(dict(
            id=f"rand{i + 1:02d}", gene="random_intergenic", locus="-",
            cls="control_random", chrom=c, strand="+", lo=lo, hi=hi,
            trunc="none", cai=None, product="random intergenic region", seq=seq,
        ))

    # ---- write -----------------------------------------------------------
    order = {"target": 0, "named": 1, "high_cai": 2, "moderate": 3,
             "control_shuffled": 4, "control_random": 5}
    records.sort(key=lambda r: (order.get(r["cls"], 9), r["id"]))

    with open(out_path, "w") as fh:
        for r in records:
            cai = f"{r['cai']:.4f}" if r["cai"] is not None else "NA"
            fh.write(
                f">{r['id']} class={r['cls']} locus={r['locus']} "
                f"chrom={r['chrom']} strand={r['strand']} "
                f"region={r['lo']}-{r['hi']} len={len(r['seq'])} "
                f"gc={gc(r['seq']):.3f} trunc={r['trunc']} cai={cai}\n"
            )
            for i in range(0, len(r["seq"]), 70):
                fh.write(r["seq"][i:i + 70] + "\n")

    man = out_path.with_suffix(out_path.suffix + ".manifest.tsv")
    cols = ["id", "gene", "locus", "cls", "chrom", "strand", "lo", "hi",
            "len", "gc", "trunc", "cai", "product"]
    with open(man, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in records:
            row = dict(r)
            row["len"] = len(r["seq"])
            row["gc"] = f"{gc(r['seq']):.4f}"
            row["cai"] = f"{r['cai']:.4f}" if r["cai"] is not None else "NA"
            fh.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")

    # ---- summary ---------------------------------------------------------
    by_cls = Counter(r["cls"] for r in records)
    trunc_n = sum(1 for r in records if r["trunc"] == "neighbour")
    log("")
    log(f"wrote {out_path}  ({len(records)} sequences)")
    log(f"      {man}")
    for c in sorted(by_cls, key=lambda k: order.get(k, 9)):
        sub = [r for r in records if r["cls"] == c]
        ls = sorted(len(r["seq"]) for r in sub)
        log(f"  {c:18s} n={len(sub):3d}  len median={ls[len(ls) // 2]:5d} "
            f"min={ls[0]:5d} max={ls[-1]:5d}  "
            f"gc={sum(gc(r['seq']) for r in sub) / len(sub):.3f}")
    log(f"  clipped at an upstream neighbour: {trunc_n}/{len(records)}")
    if tgt_gid:
        t = next((r for r in records if r["cls"] == "target"), None)
        if t:
            log(f"  target promoter: {t['id']} len={len(t['seq'])} "
                f"gc={gc(t['seq']):.3f} trunc={t['trunc']}")


if __name__ == "__main__":
    main()