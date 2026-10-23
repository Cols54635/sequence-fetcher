"""
Sequence Fetcher
----------------
Streamlit app that pulls nucleotide and protein sequences for a gene and reports
length, GC content, protein molecular weight and pI.

Data sources (both free, no key needed):
  * UniProt REST  - protein records, isoforms, signal peptides, chains, domains
  * NCBI E-utilities - RefSeq mRNA (GenBank format, for CDS / UTR boundaries)

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import io
import re
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from Bio import Align, SeqIO
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis

st.set_page_config(page_title="Sequence Fetcher", page_icon="🧬", layout="wide")

UNIPROT = "https://rest.uniprot.org/uniprotkb"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": "sequence-fetcher-streamlit/1.0"}

SPECIES = {
    "Human (Homo sapiens)": 9606,
    "Mouse (Mus musculus)": 10090,
    "Rat (Rattus norvegicus)": 10116,
    "Zebrafish (Danio rerio)": 7955,
    "Cow (Bos taurus)": 9913,
    "Pig (Sus scrofa)": 9823,
    "Rhesus macaque (Macaca mulatta)": 9544,
    "Chicken (Gallus gallus)": 9031,
    "Other (enter NCBI taxon ID)": 0,
}
HUMAN, MOUSE = "Human (Homo sapiens)", "Mouse (Mus musculus)"

# Shown in the About box. Put an address between the quotes to display a contact line.
CONTACT_EMAIL = ""
LAB = "Woeller Lab, Flaum Eye Institute, University of Rochester Medical Center"

ACC_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


# --------------------------------------------------------------------------- #
# Networking
# --------------------------------------------------------------------------- #
def http_get(url: str, params: dict | None = None, tries: int = 3) -> requests.Response:
    err = "no response"
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException as exc:
            err = str(exc)
        else:
            if r.status_code == 200:
                return r
            err = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request to {url} failed ({err}).")


@st.cache_data(ttl=86400, show_spinner=False)
def uniprot_search(query: str) -> list:
    r = http_get(f"{UNIPROT}/search", {"query": query, "format": "json", "size": 10})
    return r.json().get("results", [])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_uniprot_fasta(acc: str) -> str:
    r = http_get(f"{UNIPROT}/{acc}.fasta")
    return "".join(r.text.splitlines()[1:]).strip()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_genbank(acc: str) -> dict:
    """Fetch a RefSeq mRNA record and split it into mRNA / 5'UTR / CDS / 3'UTR."""
    r = http_get(
        f"{EUTILS}/efetch.fcgi",
        {"db": "nuccore", "id": acc, "rettype": "gb", "retmode": "text", "tool": "seq_fetcher"},
    )
    rec = SeqIO.read(io.StringIO(r.text), "genbank")
    out = {"acc": rec.id, "desc": rec.description, "mrna": str(rec.seq).upper()}
    out["exons"] = sorted({(int(f.location.start), int(f.location.end)) for f in rec.features if f.type == "exon"})
    out["polya"] = [
        (f.type, int(f.location.start), int(f.location.end))
        for f in rec.features
        if f.type in ("polyA_signal", "polyA_site")
    ]
    cds = next((f for f in rec.features if f.type == "CDS"), None)
    if cds is not None:
        out["cds_start"] = int(cds.location.start)
        out["cds_end"] = int(cds.location.end)
        out["cds"] = str(cds.location.extract(rec.seq)).upper()
    return out


def find_entries(term: str, taxon: int) -> list:
    term = term.strip()
    if not re.fullmatch(r"[\w.\-]{1,40}", term):
        return []
    if ACC_RE.match(term.upper()):
        queries = [f"accession:{term.upper()}"]
    else:
        queries = [
            f'gene_exact:"{term}" AND organism_id:{taxon} AND reviewed:true',
            f'gene_exact:"{term}" AND organism_id:{taxon}',
            f'gene:"{term}" AND organism_id:{taxon}',
        ]
    for q in queries:
        res = uniprot_search(q)
        if res:
            return res
    return []


# --------------------------------------------------------------------------- #
# UniProt entry helpers
# --------------------------------------------------------------------------- #
def gene_symbol(e: dict) -> str:
    genes = e.get("genes", [])
    return genes[0].get("geneName", {}).get("value", "") if genes else ""


def protein_name(e: dict) -> str:
    d = e.get("proteinDescription", {})
    rn = d.get("recommendedName") or (d.get("submissionNames") or [{}])[0]
    return rn.get("fullName", {}).get("value", "")


def organism_label(e: dict) -> str:
    o = e.get("organism", {})
    name = o.get("scientificName", "")
    common = o.get("commonName")
    return f"{name} ({common})" if common else name


def list_isoforms(e: dict) -> list:
    out = []
    for c in e.get("comments", []):
        if c.get("commentType") == "ALTERNATIVE PRODUCTS":
            for iso in c.get("isoforms", []):
                ids = iso.get("isoformIds", [])
                if ids:
                    out.append(
                        {
                            "id": ids[0],
                            "name": iso.get("name", {}).get("value", ""),
                            "status": iso.get("isoformSequenceStatus", ""),
                        }
                    )
    if not out:
        out = [{"id": e["primaryAccession"], "name": "single isoform", "status": "Displayed"}]
    return out


def canonical_id(isos: list) -> str:
    return next((i["id"] for i in isos if i["status"] == "Displayed"), isos[0]["id"])


def refseq_for(e: dict, iso_id: str, is_canonical: bool):
    pairs = []
    for x in e.get("uniProtKBCrossReferences", []):
        if x.get("database") != "RefSeq":
            continue
        nm = next(
            (p["value"] for p in x.get("properties", []) if p.get("key") == "NucleotideSequenceId"),
            None,
        )
        pairs.append((x.get("isoformId"), x.get("id"), nm))
    exact = [p for p in pairs if p[0] == iso_id]
    if exact:
        return exact[0][1], exact[0][2]
    if is_canonical:
        plain = [p for p in pairs if p[0] is None]
        pick = plain or pairs
        if pick:
            return pick[0][1], pick[0][2]
    return None, None


def xrefs(e: dict, db: str) -> list:
    return [x for x in e.get("uniProtKBCrossReferences", []) if x.get("database") == db]


def feats(e: dict, types: set) -> list:
    out = []
    for f in e.get("features", []):
        if f.get("type") in types:
            loc = f.get("location", {})
            s = loc.get("start", {}).get("value")
            en = loc.get("end", {}).get("value")
            if isinstance(s, int) and isinstance(en, int):
                out.append((f["type"], s, en, f.get("description", "")))
    return out


def load_record(entry: dict, iso: dict, isos: list) -> dict:
    is_canon = iso["id"] == canonical_id(isos)
    warn: list = []
    if is_canon:
        seq = entry["sequence"]["value"]
    else:
        try:
            seq = fetch_uniprot_fasta(iso["id"])
        except RuntimeError as exc:
            return {"seq": "", "np": None, "nm": None, "gb": None, "canonical": False, "warn": [str(exc)]}
    np_id, nm = refseq_for(entry, iso["id"], is_canon)
    gb = None
    if nm:
        try:
            gb = fetch_genbank(nm)
        except Exception as exc:  # network or parse problems should not kill the page
            warn.append(f"Could not retrieve {nm} from NCBI: {exc}")
    else:
        warn.append("UniProt lists no RefSeq mRNA for this isoform, so no nucleotide sequence is shown.")
    return {"seq": seq, "np": np_id, "nm": nm, "gb": gb, "canonical": is_canon, "warn": warn}


# --------------------------------------------------------------------------- #
# Calculations
# --------------------------------------------------------------------------- #
def gc_percent(seq: str) -> float:
    return gc_fraction(seq) * 100 if seq else 0.0


def protein_stats(seq: str) -> dict:
    clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())
    if not clean:
        return {"mw": float("nan"), "pi": float("nan"), "skipped": len(seq)}
    pa = ProteinAnalysis(clean)
    return {
        "mw": pa.molecular_weight() / 1000,
        "pi": pa.isoelectric_point(),
        "skipped": len(seq) - len(clean),
    }


def region_rows(entry: dict, seq: str) -> list:
    """Full-length, mature, and per-chain property rows for the canonical sequence."""
    rows = [("Full-length precursor", f"1-{len(seq)}", seq)]
    mask = [True] * len(seq)
    for _, s, e, _ in feats(entry, {"Signal", "Propeptide"}):
        for i in range(max(s - 1, 0), min(e, len(seq))):
            mask[i] = False
    mature = "".join(c for c, keep in zip(seq, mask) if keep)
    if len(mature) != len(seq):
        kept = [i + 1 for i, k in enumerate(mask) if k]
        contiguous = kept and kept[-1] - kept[0] + 1 == len(kept)
        span = f"{kept[0]}-{kept[-1]}" if contiguous else f"{len(mature)} residues (non-contiguous)"
        rows.append(("Mature (signal peptide / propeptide removed)", span, mature))
    for _, s, e, d in feats(entry, {"Chain"}):
        if e > len(seq):
            continue
        sub = seq[s - 1 : e]
        if sub in (seq, mature):
            continue
        rows.append((f"Chain: {d or 'unnamed'}", f"{s}-{e}", sub))
    return rows


def fasta(header: str, seq: str, width: int = 70) -> str:
    return f">{header}\n" + "\n".join(seq[i : i + width] for i in range(0, len(seq), width)) + "\n"


def prot_aligner() -> Align.PairwiseAligner:
    a = Align.PairwiseAligner()
    a.mode = "global"
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score = -10
    a.extend_gap_score = -0.5
    return a


def nuc_aligner() -> Align.PairwiseAligner:
    a = Align.PairwiseAligner()
    a.mode = "global"
    a.match_score = 2
    a.mismatch_score = -3
    a.open_gap_score = -5
    a.extend_gap_score = -2
    return a


def identity(a: str, b: str, aligner) -> tuple:
    aln = aligner.align(a, b)[0]
    matches = aligned_cols = 0
    for (as_, ae), (bs, be) in zip(*aln.aligned):
        as_, ae, bs, be = int(as_), int(ae), int(bs), int(be)
        aligned_cols += ae - as_
        matches += sum(1 for x, y in zip(a[as_:ae], b[bs:be]) if x == y)
    columns = len(a) + len(b) - aligned_cols
    return matches / columns * 100, matches, columns, aln


def clean_protein(seq: str) -> str:
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", seq.upper().replace("U", "C"))


# --------------------------------------------------------------------------- #
# mRNA map, oligo and peptide placement
# --------------------------------------------------------------------------- #
STD_AA = "ACDEFGHIKLMNPQRSTVWY"


def esc_html(t: str) -> str:
    return t.replace("<", "&lt;").replace(">", "&gt;")


def revcomp(s: str) -> str:
    return str(Seq(s).reverse_complement())


def parse_oligos(text: str) -> list:
    """Accepts 'name sequence' lines, bare sequences, or FASTA records."""
    out, pending = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            pending = line[1:].strip() or None
            continue
        tokens = [t for t in re.split(r"[\s:,;]+", line) if t]
        seq = re.sub(r"[^ACGTUNacgtun]", "", tokens[-1]).upper().replace("U", "T")
        name = " ".join(tokens[:-1]) or pending or f"Oligo {len(out) + 1}"
        pending = None
        if seq:
            out.append((name, seq))
    return out


def parse_peptides(text: str) -> list:
    """Accepts 'name peptide' lines, bare peptides, or FASTA records (one-letter codes)."""
    out, pending = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            pending = line[1:].strip() or None
            continue
        tokens = [t for t in re.split(r"[\s:,;]+", line) if t]
        pep = re.sub(f"[^{STD_AA}]", "", re.sub(r"[^A-Za-z]", "", tokens[-1]).upper())
        name = " ".join(tokens[:-1]) or pending or f"Peptide {len(out) + 1}"
        pending = None
        if pep:
            out.append((name, pep))
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def find_oligo(mrna: str, name: str, oligo: str, max_mm: int) -> list:
    """Find an oligo on the transcript, same sense (forward) or reverse complement (reverse)."""
    hits = []
    n = len(oligo)
    for orientation, q in (("forward", oligo), ("reverse", revcomp(oligo))):
        for i in range(len(mrna) - n + 1):
            mm = 0
            for a, b in zip(mrna[i : i + n], q):
                if a != b and b != "N":
                    mm += 1
                    if mm > max_mm:
                        break
            if mm <= max_mm:
                hits.append(
                    {"name": name, "orientation": orientation, "start": i + 1, "end": i + n, "mismatches": mm}
                )
    return hits


@st.cache_data(ttl=3600, show_spinner=False)
def find_peptide(prot: str, name: str, pep: str, max_mm: int) -> list:
    """Find a peptide in the protein translated from this transcript (1-based residues)."""
    hits, n = [], len(pep)
    for i in range(len(prot) - n + 1):
        mm = 0
        for x, y in zip(prot[i : i + n], pep):
            if x != y:
                mm += 1
                if mm > max_mm:
                    break
        if mm <= max_mm:
            hits.append({"name": name, "start": i + 1, "end": i + n, "mismatches": mm})
    return hits


def annotate_span(entry: dict, start: int, end: int) -> dict:
    """What UniProt annotates under residues start-end (1-based, inclusive)."""

    def over(types):
        return [x for x in feats(entry, types) if x[1] <= end and start <= x[2]]

    topo = sorted({x[3] for x in over({"Topological domain"}) if x[3]})
    tm = over({"Transmembrane"})
    sig = over({"Signal"})
    doms = [x[3] or x[0] for x in over({"Domain", "Repeat"})]
    chains_all = feats(entry, {"Chain", "Peptide"})
    chains = [x[3] or "Chain" for x in chains_all if x[1] <= end and start <= x[2]]
    within = (
        ("Yes" if any(x[1] <= start and end <= x[2] for x in chains_all) else "No") if chains_all else "n/a"
    )
    return {
        "Topology": "; ".join(topo) or "n/a",
        "Transmembrane": f"Yes ({len(tm)} segment{'s' if len(tm) != 1 else ''})" if tm else "No",
        "Signal peptide": "Overlaps" if sig else "No",
        "Domain(s)": "; ".join(doms) or "-",
        "Chain(s)": "; ".join(chains) or "-",
        "Within one chain": within,
    }


def region_name(pos0: int, c0: int, c1: int) -> str:
    return "5' UTR" if pos0 < c0 else ("CDS" if pos0 < c1 else "3' UTR")


def region_span(a0: int, b0: int, c0: int, c1: int) -> str:
    r1, r2 = region_name(a0, c0, c1), region_name(b0 - 1, c0, c1)
    return r1 if r1 == r2 else f"{r1} / {r2}"


def map_figure(gb: dict, entry: dict, prot: str, show_protein: bool, hits: list, pep_hits: list, on: dict):
    """Scaled transcript diagram. `on` says which optional rows to draw."""
    n = len(gb["mrna"])
    c0, c1 = gb["cds_start"], gb["cds_end"]
    rows = []  # (label, [(x0, x1, color, hover, half_height)])

    parts = [
        (0, c0, "#8a8f98", f"5' UTR: 1-{c0} ({c0:,} nt)", 0.16),
        (c0, c1, "#4c78a8", f"CDS: {c0 + 1}-{c1} ({c1 - c0:,} nt)", 0.32),
        (c1, n, "#8a8f98", f"3' UTR: {c1 + 1}-{n} ({n - c1:,} nt)", 0.16),
    ]
    rows.append(("Transcript", [p for p in parts if p[1] > p[0]]))

    if on.get("exons") and gb.get("exons"):
        items = []
        for k, (a, b) in enumerate(gb["exons"], start=1):
            items.append((a, b, "#4c78a8" if k % 2 else "#86a8cf", f"Exon {k}: {a + 1}-{b} ({b - a:,} nt)", 0.28))
        rows.append(("Exons", items))

    if show_protein:
        layers = [
            ("signal", "Signal peptide", {"Signal"}, "#e0a030"),
            ("chains", "Chains", {"Chain"}, "#7b6fb5"),
            ("tm", "Transmembrane", {"Transmembrane"}, "#d1495b"),
            ("domains", "Domains", {"Domain", "Repeat"}, "#3aa17e"),
            ("glyc", "Glycosylation", {"Glycosylation"}, "#2b8cbe"),
            ("mods", "Modified residues", {"Modified residue"}, "#e377c2"),
        ]
        for key, label, types, color in layers:
            if not on.get(key):
                continue
            items = []
            for t, s, e, d in feats(entry, types):
                head = f"{prot[s - 1]}{s}" if s == e and s <= len(prot) else f"aa {s}-{e}"
                hover = f"{t}: {head}" + (f"<br>{esc_html(d[:140])}" if d else "")
                items.append((c0 + (s - 1) * 3, c0 + e * 3, color, hover, 0.28))
            if items:
                rows.append((label, items))

    if on.get("polya") and gb.get("polya"):
        rows.append(
            ("PolyA", [(a, b, "#b5651d", f"{t.replace('_', ' ')}: {a + 1}-{b}", 0.28) for t, a, b in gb["polya"]])
        )

    if on.get("oligos"):
        for orient, label, color in (("forward", "Forward oligos", "#17a2b8"), ("reverse", "Reverse oligos", "#e83e8c")):
            items = [
                (
                    h["start"] - 1, h["end"], color,
                    f"{esc_html(h['name'])} ({orient})<br>{h['start']}-{h['end']}, {h['mismatches']} mismatch(es)",
                    0.28,
                )
                for h in hits
                if h["orientation"] == orient
            ]
            if items:
                rows.append((label, items))

    if on.get("peptides") and pep_hits:
        rows.append(
            (
                "Peptides",
                [
                    (
                        c0 + (h["start"] - 1) * 3, c0 + h["end"] * 3, "#9c27b0",
                        f"{esc_html(h['name'])}<br>aa {h['start']}-{h['end']}, nt {c0 + (h['start'] - 1) * 3 + 1}-"
                        f"{c0 + h['end'] * 3}, {h['mismatches']} mismatch(es)",
                        0.28,
                    )
                    for h in pep_hits
                ],
            )
        )

    fig = go.Figure()
    total = len(rows)
    wmin = n * 0.003  # keep one-nucleotide and single-residue features visible
    for idx_, (label, items) in enumerate(rows):
        y = total - idx_
        xs, hov = [], []
        for x0, x1, color, hover, half in items:
            x1 = max(x1, x0 + wmin)
            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=y - half, y1=y + half,
                fillcolor=color, line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
            )
            xs.append((x0 + x1) / 2)
            hov.append(hover)
        fig.add_trace(
            go.Scatter(
                x=xs, y=[y] * len(xs), mode="markers", marker=dict(opacity=0, size=14),
                text=hov, hoverinfo="text", showlegend=False,
            )
        )
    if on.get("codons", True):
        for x, text in ((c0, "ATG"), (c1, "stop")):
            fig.add_shape(
                type="line", x0=x, x1=x, y0=0.45, y1=total + 0.45,
                line=dict(width=1, color="rgba(128,128,128,0.8)", dash="dot"),
            )
            fig.add_annotation(x=x, y=total + 0.62, text=text, showarrow=False, font=dict(size=10))
    fig.update_xaxes(range=[0, n], title="Position in transcript (nt)")
    fig.update_yaxes(
        tickvals=[total - i for i in range(total)],
        ticktext=[r[0] for r in rows],
        range=[0.4, total + 0.95], showgrid=False, zeroline=False,
    )
    fig.update_layout(height=100 + 46 * total, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("🧬 Sequence Fetcher")
st.caption("Nucleotide and protein sequences from UniProt and NCBI RefSeq, with length, GC content, MW and pI.")

with st.expander("About this tool"):
    st.markdown(
        f"""
Built by the **{LAB}** as a research convenience.
It is not an official product of the University of Rochester.

**What it does.** Looks up a gene, shows its RefSeq mRNA (5' UTR, coding sequence, 3' UTR) and UniProt protein,
and calculates length, GC content, molecular weight and pI. It also compares mature and full-length forms,
lists isoforms, and compares orthologs between species.

**Where the data comes from.** Protein records, isoforms and annotations come from
[UniProt](https://www.uniprot.org); transcripts come from [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/refseq/).
Please cite the source databases when you use their data:
- The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase.* Nucleic Acids Research.
- O'Leary NA, et al. *Reference sequence (RefSeq) database at NCBI.* Nucleic Acids Research.

**Calculations.** MW and pI are computed with Biopython's ProtParam and can differ slightly from other tools.
Calculated MW does not account for glycosylation or other modifications, so bands on a Western blot may run larger.
Please verify important sequences against the source record before ordering primers or constructs.

**Privacy.** Only public database lookups are made, and nothing you type is stored by this app.
Avoid entering unpublished or confidential sequences.
"""
    )
    if CONTACT_EMAIL:
        st.markdown(f"**Questions or corrections:** {CONTACT_EMAIL}")

if "q" not in st.session_state:
    st.session_state.q = ("TSHR", HUMAN, 9606)

with st.form("search"):
    c1, c2, c3, c4 = st.columns([2.2, 2.2, 1.4, 1])
    term_in = c1.text_input("Gene symbol or UniProt accession", value=st.session_state.q[0])
    sp_in = c2.selectbox("Species", list(SPECIES), index=list(SPECIES).index(st.session_state.q[1]))
    custom_in = c3.text_input("Taxon ID (if Other)", value="")
    submitted = c4.form_submit_button("Fetch", type="primary")

if submitted:
    taxon_in = SPECIES[sp_in]
    if taxon_in == 0:
        taxon_in = int(custom_in) if custom_in.strip().isdigit() else 0
    if taxon_in == 0 and not ACC_RE.match(term_in.strip().upper()):
        st.error("Enter a numeric NCBI taxon ID for 'Other', or use a UniProt accession.")
    else:
        st.session_state.q = (term_in.strip(), sp_in, taxon_in)

term, sp_label, taxon = st.session_state.q

try:
    with st.spinner("Searching UniProt..."):
        entries = find_entries(term, taxon)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if not entries:
    st.warning(
        f"No UniProt entry found for '{term}' in this species. Check the symbol "
        "(mouse and rat symbols are usually capitalised differently, but search ignores case), "
        "or try a UniProt accession."
    )
    st.stop()

if len(entries) > 1:
    idx = st.selectbox(
        "Matching entries",
        range(len(entries)),
        format_func=lambda i: (
            f"{entries[i]['primaryAccession']} - {gene_symbol(entries[i])} - "
            f"{protein_name(entries[i])[:60]} ({entries[i].get('entryType', '').split(' ')[0]})"
        ),
    )
else:
    idx = 0
entry = entries[idx]
acc = entry["primaryAccession"]
isos = list_isoforms(entry)
canon = canonical_id(isos)

iso_labels = {
    i["id"]: f"{i['id']}  |  {i['name']}  |  {i['status']}" + ("  (canonical)" if i["id"] == canon else "")
    for i in isos
}
iso_id = st.selectbox(
    "Isoform",
    [i["id"] for i in isos],
    format_func=lambda x: iso_labels[x],
    help="The canonical isoform is UniProt's reference sequence. Other isoforms are fetched individually.",
)
iso = next(i for i in isos if i["id"] == iso_id)

with st.spinner("Fetching sequences from UniProt and NCBI..."):
    rec = load_record(entry, iso, isos)

if not rec["seq"]:
    for w in rec["warn"]:
        st.error(w)
    st.stop()

org = organism_label(entry)
gene = gene_symbol(entry) or term
st.subheader(f"{gene}  ·  {protein_name(entry)}")
st.caption(f"{org}  ·  UniProt {acc}  ·  isoform {iso_id}")
for w in rec["warn"]:
    st.warning(w)

gb = rec["gb"]
cds = gb.get("cds") if gb else None
pst = protein_stats(rec["seq"])

m = st.columns(6)
m[0].metric("mRNA length", f"{len(gb['mrna']):,} nt" if gb else "n/a")
m[1].metric("CDS length", f"{len(cds):,} nt" if cds else "n/a")
m[2].metric("CDS GC content", f"{gc_percent(cds):.1f}%" if cds else "n/a")
m[3].metric("Protein length", f"{len(rec['seq']):,} aa")
m[4].metric("Protein MW", f"{pst['mw']:.2f} kDa")
m[5].metric("Protein pI", f"{pst['pi']:.2f}")

tab_nt, tab_map, tab_prot, tab_mature, tab_cmp, tab_links = st.tabs(
    ["Nucleotide", "mRNA map", "Protein", "Mature vs full-length", "Species comparison", "Links"]
)

# ---- Nucleotide ----------------------------------------------------------- #
with tab_nt:
    if not gb:
        st.info("No RefSeq mRNA is available for this isoform.")
    else:
        st.caption(f"RefSeq transcript {gb['acc']} · {gb['desc']}")
        parts = [("Full mRNA (RefSeq transcript)", gb["mrna"])]
        if cds:
            parts += [
                ("5' UTR", gb["mrna"][: gb["cds_start"]]),
                ("Coding sequence (CDS)", cds),
                ("3' UTR", gb["mrna"][gb["cds_end"] :]),
            ]
        st.dataframe(
            pd.DataFrame(
                [{"Region": n, "Length (nt)": len(s), "GC content (%)": round(gc_percent(s), 2)} for n, s in parts]
            ),
            hide_index=True,
        )
        if cds:
            translated = str(Seq(cds).translate(to_stop=True))
            if translated == rec["seq"]:
                st.success("Check passed: the CDS translates to exactly the UniProt protein sequence.")
            else:
                st.warning(
                    f"The translated CDS ({len(translated)} aa) differs from the UniProt sequence "
                    f"({len(rec['seq'])} aa). This can happen with selenoproteins, RNA editing, or "
                    "when the RefSeq and UniProt records are not versions of the same isoform."
                )
            fa_cds = fasta(f"{gb['acc']} {gene} {org} CDS", cds)
            st.markdown("**Coding sequence (FASTA)**")
            st.code(fa_cds, language=None)
            st.download_button("Download CDS FASTA", fa_cds, f"{gene}_{gb['acc']}_CDS.fasta", key="dl_cds")
        fa_mrna = fasta(f"{gb['acc']} {gene} {org} mRNA", gb["mrna"])
        with st.expander("Full mRNA (FASTA)"):
            st.code(fa_mrna, language=None)
            st.download_button("Download mRNA FASTA", fa_mrna, f"{gene}_{gb['acc']}_mRNA.fasta", key="dl_mrna")

# ---- mRNA map ------------------------------------------------------------- #
with tab_map:
    if not gb or not cds:
        st.info("A RefSeq mRNA with an annotated coding sequence is needed to draw this map.")
    else:
        mrna = gb["mrna"]
        mlen = len(mrna)
        c0, c1 = gb["cds_start"], gb["cds_end"]
        exon_list = gb.get("exons", [])
        junctions = [end for _, end in exon_list[:-1]]
        cds_protein = str(Seq(cds).translate(to_stop=True))
        translation_ok = cds_protein == rec["seq"]
        show_protein = rec["canonical"] and translation_ok

        with st.expander("Choose what appears on the map and in the PNG"):
            st.caption("Untick a row to remove it from the figure. The camera icon saves exactly what is shown.")
            on = {
                "exons": st.checkbox("Exons", value=True, key="map_exons"),
                "signal": st.checkbox("Signal peptide", value=True, key="map_signal"),
                "chains": st.checkbox("Chains", value=True, key="map_chains"),
                "tm": st.checkbox("Transmembrane segments", value=True, key="map_tm"),
                "domains": st.checkbox("Domains", value=True, key="map_domains"),
                "glyc": st.checkbox("Glycosylation sites", value=False, key="map_glyc"),
                "mods": st.checkbox("Modified residues (for example phosphorylation)", value=False, key="map_mods"),
                "polya": st.checkbox("PolyA signal and site", value=True, key="map_polya"),
                "oligos": st.checkbox("Primer and siRNA matches", value=True, key="map_oligos"),
                "peptides": st.checkbox("Peptide matches", value=True, key="map_peptides"),
                "codons": st.checkbox("Start and stop codon lines", value=True, key="map_codons"),
            }

        with st.expander("Mark primers, siRNAs or peptides (optional)"):
            oligo_text = st.text_area(
                "Oligo sequences (DNA or RNA, 5' to 3')",
                height=100,
                key="oligo_text",
                placeholder="One per line, for example:\nTSHR-F  ACGTACGTACGTACGTACGT\nTSHR-R  TGCATGCATGCATGCATGCA",
                help="Name and sequence on each line, a bare sequence, or FASTA. U is treated as T.",
            )
            max_mm = st.slider("Allowed mismatches (oligos)", 0, 3, 0, key="oligo_mm")
            pep_text = st.text_area(
                "Peptide sequences (protein, one-letter codes)",
                height=100,
                key="pep_text",
                placeholder="One per line, for example:\nImmunogen  ACDEFGHIKLMNPQ\nTryptic-peptide  LVNEVTEFAK",
                help="Antibody immunogens, epitopes or mass-spec peptides. They are located on the protein "
                "translated from this transcript, then drawn on the CDS.",
            )
            pep_mm = st.slider("Allowed mismatches (peptides)", 0, 2, 0, key="pep_mm")

        hits, notes = [], []
        for oname, oseq in parse_oligos(oligo_text):
            if len(oseq) < 12:
                notes.append(f"{oname}: shorter than 12 nt, so it was skipped (too short to place reliably).")
                continue
            found = find_oligo(mrna, oname, oseq, max_mm)
            if not found:
                notes.append(
                    f"{oname}: no match on this transcript. It may target a different isoform or species, "
                    "or you can allow a mismatch."
                )
            hits += found

        pep_hits = []
        for pname, pep in parse_peptides(pep_text):
            if len(pep) < 5:
                notes.append(f"{pname}: shorter than 5 residues, so it was skipped.")
                continue
            found = find_peptide(cds_protein, pname, pep, pep_mm)
            if not found:
                notes.append(
                    f"{pname}: no match in the protein encoded by this transcript. It may belong to another "
                    "isoform or species, or you can allow a mismatch."
                )
            pep_hits += found

        st.plotly_chart(
            map_figure(gb, entry, cds_protein, show_protein, hits, pep_hits, on),
            config={"toImageButtonOptions": {"format": "png", "filename": f"{gene}_mRNA_map", "scale": 3}},
        )
        st.caption(
            f"{gb['acc']} · {mlen:,} nt. Drawn to scale. Hover over a feature for its coordinates. "
            "The camera icon at the top right of the figure saves a high-resolution PNG of what is shown."
        )
        if not rec["canonical"]:
            st.caption("Protein features (signal peptide, chains, domains, sites) are shown only for the canonical isoform.")
        elif not translation_ok:
            st.caption("Protein features are hidden because the RefSeq CDS does not translate to the UniProt sequence.")
        if not exon_list:
            st.caption("This RefSeq record does not list exon boundaries, so exons are not drawn.")
        for note in notes:
            st.warning(note)

        regions = []
        if c0 > 0:
            regions.append({"Region": "5' UTR", "Start": 1, "End": c0, "Length (nt)": c0})
        regions.append({"Region": "CDS", "Start": c0 + 1, "End": c1, "Length (nt)": c1 - c0})
        if mlen > c1:
            regions.append({"Region": "3' UTR", "Start": c1 + 1, "End": mlen, "Length (nt)": mlen - c1})
        st.dataframe(pd.DataFrame(regions), hide_index=True)

        if exon_list:
            with st.expander(f"Exons ({len(exon_list)})"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Exon": k,
                                "Start": a + 1,
                                "End": b,
                                "Length (nt)": b - a,
                                "Region": region_span(a, b, c0, c1),
                            }
                            for k, (a, b) in enumerate(exon_list, start=1)
                        ]
                    ),
                    hide_index=True,
                )

        if hits:
            st.markdown("**Oligo matches**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Oligo": h["name"],
                            "Orientation": "Forward (same as transcript)"
                            if h["orientation"] == "forward"
                            else "Reverse (complement of transcript)",
                            "Start": h["start"],
                            "End": h["end"],
                            "Region": region_span(h["start"] - 1, h["end"], c0, c1),
                            "Mismatches": h["mismatches"],
                            "Spans exon junction": (
                                "Yes" if any(h["start"] - 1 < j < h["end"] for j in junctions) else "No"
                            )
                            if exon_list
                            else "n/a",
                        }
                        for h in hits
                    ]
                ),
                hide_index=True,
            )
            fwd = [h for h in hits if h["orientation"] == "forward"]
            rev = [h for h in hits if h["orientation"] == "reverse"]
            amps = []
            for f_ in fwd:
                for r_ in rev:
                    if r_["end"] > f_["start"]:
                        size = r_["end"] - f_["start"] + 1
                        spans = any(f_["start"] - 1 < j < r_["end"] for j in junctions)
                        amps.append(
                            {
                                "Forward": f_["name"],
                                "Reverse": r_["name"],
                                "Amplicon (bp)": size,
                                "Region": region_span(f_["start"] - 1, r_["end"], c0, c1),
                                "Crosses exon junction": ("Yes" if spans else "No") if exon_list else "n/a",
                            }
                        )
            if amps:
                st.markdown("**Possible amplicons** (forward oligo upstream of a reverse oligo)")
                st.dataframe(pd.DataFrame(amps[:12]), hide_index=True)
            st.caption(
                "Matches are placed on the transcript sequence only. Check specificity separately "
                "(for example with Primer-BLAST, linked in the Links tab) before ordering oligos."
            )

        if pep_hits:
            st.markdown("**Peptide matches**")
            pep_rows = []
            for h in pep_hits:
                a0, b0 = c0 + (h["start"] - 1) * 3, c0 + h["end"] * 3
                row = {
                    "Peptide": h["name"],
                    "Residues": f"{h['start']}-{h['end']}",
                    "Transcript position (nt)": f"{a0 + 1}-{b0}",
                    "Mismatches": h["mismatches"],
                    "Spans exon junction": ("Yes" if any(a0 < j < b0 for j in junctions) else "No")
                    if exon_list
                    else "n/a",
                }
                if show_protein:
                    row.update(annotate_span(entry, h["start"], h["end"]))
                pep_rows.append(row)
            st.dataframe(pd.DataFrame(pep_rows), hide_index=True)
            st.caption(
                "Annotation columns come from UniProt for the canonical isoform. 'Within one chain: No' means the "
                "peptide crosses a cleavage site or the signal peptide boundary."
            )

# ---- Protein -------------------------------------------------------------- #
with tab_prot:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Isoform": iso_id,
                    "Length (aa)": len(rec["seq"]),
                    "MW (kDa)": round(pst["mw"], 2),
                    "pI": round(pst["pi"], 2),
                }
            ]
        ),
        hide_index=True,
    )
    if pst["skipped"]:
        st.caption(f"{pst['skipped']} non-standard residue(s) were left out of the MW and pI calculation.")
    fa_prot = fasta(f"{iso_id} {protein_name(entry)} OS={org}", rec["seq"])
    st.code(fa_prot, language=None)
    st.download_button("Download protein FASTA", fa_prot, f"{gene}_{iso_id}.fasta", key="dl_prot")
    st.caption(
        "MW and pI are calculated with Biopython's ProtParam. They can differ slightly from the ExPASy web "
        "tool because the pKa scales differ."
    )

# ---- Mature vs full-length ------------------------------------------------ #
with tab_mature:
    if not rec["canonical"]:
        st.info(
            "UniProt's signal peptide, chain and domain positions are defined on the canonical isoform, "
            "so only the full-length values are shown for this isoform. Select the canonical isoform to "
            "see the mature and chain breakdown."
        )
        rows = [("Full-length (selected isoform)", f"1-{len(rec['seq'])}", rec["seq"])]
    else:
        rows = region_rows(entry, rec["seq"])
    table = []
    for label, span, s in rows:
        st_ = protein_stats(s)
        table.append(
            {
                "Region": label,
                "Residues": span,
                "Length (aa)": len(s),
                "MW (kDa)": round(st_["mw"], 2),
                "pI": round(st_["pi"], 2),
            }
        )
    st.dataframe(pd.DataFrame(table), hide_index=True)

    labels = [r[0] for r in rows]
    pick = st.selectbox(
        "Show sequence for",
        labels,
        index=1 if len(labels) > 1 else 0,
        help="Pick the mature protein or an individual chain to copy or download its sequence.",
    )
    r_label, r_span, r_seq = rows[labels.index(pick)]
    r_header = f"{iso_id} {protein_name(entry)} | {r_label} | residues {r_span} | OS={org}"
    r_fa = fasta(r_header, r_seq)
    r_file = re.sub(r"[^A-Za-z0-9]+", "_", r_label)[:40].strip("_")
    st.code(r_fa, language=None)
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download this sequence (FASTA)", r_fa, f"{gene}_{r_file}.fasta", key="dl_region")
    all_fa = "".join(
        fasta(f"{iso_id} {protein_name(entry)} | {lbl} | residues {sp} | OS={org}", sq) for lbl, sp, sq in rows
    )
    dl2.download_button("Download all regions (FASTA)", all_fa, f"{gene}_all_regions.fasta", key="dl_regions_all")

    if len(rows) == 1 and rec["canonical"]:
        st.caption("UniProt has no signal peptide, propeptide or chain annotation for this entry.")
    if rec["canonical"]:
        st.caption("Signal peptide, chains, transmembrane segments and domains are drawn to scale on the mRNA map tab.")
    st.caption(
        "Calculated values assume no modifications. Glycosylation, phosphorylation and cleavage steps "
        "make bands on a Western blot run larger than the calculated MW."
    )

# ---- Species comparison --------------------------------------------------- #
with tab_cmp:
    st.markdown(f"Compare the canonical **{gene}** from {org} with another species.")
    options = [k for k in SPECIES if SPECIES[k] != 0 and SPECIES[k] != entry.get("organism", {}).get("taxonId")]
    default_b = MOUSE if MOUSE in options else options[0]
    cc1, cc2 = st.columns(2)
    sp_b = cc1.selectbox("Compare with", options, index=options.index(default_b))
    sym_b = cc2.text_input("Gene symbol in that species (if different)", value="")
    term_b = sym_b.strip() or gene
    try:
        with st.spinner("Fetching the second species..."):
            res_b = find_entries(term_b, SPECIES[sp_b])
            if not res_b:
                st.warning(f"No UniProt entry found for '{term_b}' in {sp_b}. Try the symbol used in that species.")
            else:
                entry_b = res_b[0]
                isos_b = list_isoforms(entry_b)
                canon_b = next(i for i in isos_b if i["id"] == canonical_id(isos_b))
                rec_b = load_record(entry_b, canon_b, isos_b)
                rec_a = (
                    rec
                    if rec["canonical"]
                    else load_record(entry, next(i for i in isos if i["id"] == canon), isos)
                )

                pa_ = clean_protein(rec_a["seq"])
                pb_ = clean_protein(rec_b["seq"])
                p_id, p_match, p_cols, p_aln = identity(pa_, pb_, prot_aligner())

                n_id = None
                cds_a = rec_a["gb"].get("cds") if rec_a["gb"] else None
                cds_b = rec_b["gb"].get("cds") if rec_b["gb"] else None
                if cds_a and cds_b:
                    if len(cds_a) * len(cds_b) <= 30_000_000:
                        n_id = identity(cds_a, cds_b, nuc_aligner())[0]

                k = st.columns(3)
                k[0].metric("Protein identity", f"{p_id:.1f}%")
                k[1].metric("CDS identity", f"{n_id:.1f}%" if n_id is not None else "n/a")
                k[2].metric("Aligned columns", f"{p_cols:,}")

                def summary(label, ent, r):
                    s = protein_stats(r["seq"])
                    g = r["gb"]
                    return {
                        "Species": label,
                        "UniProt": ent["primaryAccession"],
                        "Protein (aa)": len(r["seq"]),
                        "MW (kDa)": round(s["mw"], 2),
                        "pI": round(s["pi"], 2),
                        "CDS (nt)": len(g["cds"]) if g and g.get("cds") else None,
                        "CDS GC (%)": round(gc_percent(g["cds"]), 1) if g and g.get("cds") else None,
                        "RefSeq mRNA": r["nm"] or "n/a",
                    }

                st.dataframe(
                    pd.DataFrame(
                        [summary(organism_label(entry), entry, rec_a), summary(organism_label(entry_b), entry_b, rec_b)]
                    ),
                    hide_index=True,
                )
                st.caption(
                    "Identity = identical residues / total alignment columns (gaps included). "
                    "Protein: global alignment, BLOSUM62. CDS: global nucleotide alignment, approximate."
                )
                with st.expander("Protein alignment"):
                    st.code(str(p_aln), language=None)
                fa_both = fasta(f"{acc} {gene} {organism_label(entry)}", rec_a["seq"]) + fasta(
                    f"{entry_b['primaryAccession']} {gene_symbol(entry_b)} {organism_label(entry_b)}", rec_b["seq"]
                )
                st.markdown("**Both proteins as one FASTA** (ready to paste into an aligner)")
                st.code(fa_both, language=None)
                st.download_button("Download comparison FASTA", fa_both, f"{gene}_comparison.fasta", key="dl_cmp")
    except RuntimeError as exc:
        st.error(str(exc))

# ---- Links ---------------------------------------------------------------- #
with tab_links:
    links = [f"- [UniProt entry {acc}](https://www.uniprot.org/uniprotkb/{acc}/entry)"]
    links.append(f"- [AlphaFold predicted structure](https://alphafold.ebi.ac.uk/entry/{acc})")
    for x in xrefs(entry, "GeneID")[:1]:
        links.append(f"- [NCBI Gene {x['id']}](https://www.ncbi.nlm.nih.gov/gene/{x['id']})")
    for x in xrefs(entry, "Ensembl")[:1]:
        gid = next((p["value"] for p in x.get("properties", []) if p.get("key") == "GeneId"), None)
        if gid:
            links.append(f"- [Ensembl gene {gid.split('.')[0]}](https://www.ensembl.org/id/{gid.split('.')[0]})")
    if rec["nm"]:
        links.append(f"- [RefSeq mRNA {rec['nm']} (NCBI)](https://www.ncbi.nlm.nih.gov/nuccore/{rec['nm']})")
        links.append(
            f"- [Primer-BLAST on {rec['nm']}](https://www.ncbi.nlm.nih.gov/tools/primer-blast/index.cgi?INPUT_SEQUENCE={rec['nm']})"
        )
    if rec["np"]:
        links.append(f"- [RefSeq protein {rec['np']} (NCBI)](https://www.ncbi.nlm.nih.gov/protein/{rec['np']})")
    st.markdown("\n".join(links))
    pdbs = xrefs(entry, "PDB")
    if pdbs:
        st.markdown(f"**Experimental structures in the PDB ({len(pdbs)})**")
        lines = []
        for x in pdbs[:12]:
            props = {p["key"]: p["value"] for p in x.get("properties", [])}
            detail = ", ".join(v for v in (props.get("Method"), props.get("Resolution")) if v)
            lines.append(f"- [{x['id']}](https://www.rcsb.org/structure/{x['id']})" + (f" - {detail}" if detail else ""))
        st.markdown("\n".join(lines))
        if len(pdbs) > 12:
            st.caption(f"Showing 12 of {len(pdbs)}. The UniProt entry lists them all.")
    else:
        st.caption("No experimental structures are listed for this entry. The AlphaFold model may still be available.")

st.caption("Data from UniProt and NCBI RefSeq. Public data only; nothing you type is stored. See 'About this tool' at the top.")
