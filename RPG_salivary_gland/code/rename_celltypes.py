"""Clearer source-gland cell-type nomenclature for the combined salivary reference.
no suffix -> (SMG)   [Hauser iScience];  (mix) -> (SMG/SLG)  [Huang GSE216476];
(parotid) -> (Parotid) [Rheinheimer GSE223516];  (SLG) kept."""

def newname(ct):
    ct = str(ct)
    if ct.endswith("(mix)"):     return ct[:-5].rstrip() + " (SMG/SLG)"
    if ct.endswith("(parotid)"): return ct[:-9].rstrip() + " (Parotid)"
    if ct.endswith("(SLG)"):     return ct
    return ct + " (SMG)"

def rename_lineage(lineage_dict):
    return {L: [newname(c) for c in cs] for L, cs in lineage_dict.items()}
