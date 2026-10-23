# mRNA and Protein Sequence Catcher

A free web tool for retrieving gene sequences and basic properties.


## What it does
- Looks up a gene by symbol or UniProt accession (human, mouse, rat, zebrafish and other species)
- Shows the RefSeq mRNA split into 5' UTR, coding sequence and 3' UTR, with length and GC content
- Shows the UniProt protein sequence with length, molecular weight and pI
- Lists isoforms and gives each its own sequence and RefSeq transcript
- Compares full-length, mature (signal peptide removed) and individual chain forms
- Compares orthologs between species (percent identity)
- Links to UniProt, AlphaFold, NCBI, Ensembl, Primer-BLAST and PDB

## Data sources
UniProt (protein records and annotations) and NCBI RefSeq (transcripts).
Please cite the source databases when you use their data.

## Notes
- MW and pI are calculated with Biopython and may differ slightly from other tools.
- Only public database lookups are made. Nothing entered is stored.
- This is a research convenience tool, not an official University of Rochester product.
  Verify important sequences against the source records.

## Contact
Woeller Lab, Flaum Eye Institute, University of Rochester Medical Center

## https://sequence-fetcher-9xlzjdttxcadmmopwvddjs.streamlit.app/
## Run locally
pip install -r requirements.txt
streamlit run app.py
