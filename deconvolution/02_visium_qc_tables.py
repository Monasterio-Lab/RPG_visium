"""Visium QC metric tables for A1 (ORES) and D1 (SMG/SLG) as one Excel workbook.
Space Ranger summary metrics are parsed from A1's web_summary.html (Space Ranger
1.2.1 has no metrics_summary.csv) and from D1's metrics_summary.csv; per-spot QC
(mean genes/UMIs, mitochondrial fraction, UMI CV) is computed from the filtered
matrices. Sheets: 'A1 (ORES)', 'D1 (SMG-SLG)', 'A1 vs D1'. Formatted for copy-paste
into Affinity. Run: ./.venv/bin/python analysis/02_visium_qc_tables.py"""
import re, sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import numpy as np, pandas as pd, scanpy as sc
from visium_io import load_visium_a1, load_visium
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

html = open("VISIUM_outs/V10S15-395_A1/web_summary.html", encoding="utf-8", errors="ignore").read()
a1raw = {}
for name, val in re.findall(r'\["([^"]+)", "([\d.,%]+)"\]', html): a1raw.setdefault(name, val)
for val, name in re.findall(r'"metric": "([\d.,%]+)", "name": "([^"]+)"', html): a1raw.setdefault(name, val)
def num(s):
    s = str(s).strip()
    return float(s[:-1].replace(",", "")) / 100.0 if s.endswith("%") else float(s.replace(",", ""))
d1csv = pd.read_csv("SMG:SLG_Outs/D1/outs/metrics_summary.csv"); d1raw = {c: d1csv.iloc[0][c] for c in d1csv.columns}

SPEC = [("Number of Spots Under Tissue","int"),("Number of Reads","int"),("Mean Reads per Spot","int"),
 ("Mean Reads Under Tissue per Spot","int"),("Fraction of Spots Under Tissue","pct"),("Median Genes per Spot","int"),
 ("Median UMI Counts per Spot","int"),("Total Genes Detected","int"),("Valid Barcodes","pct"),("Valid UMIs","pct"),
 ("Sequencing Saturation","pct"),("Q30 Bases in Barcode","pct"),("Q30 Bases in RNA Read","pct"),("Q30 Bases in UMI","pct"),
 ("Reads Mapped to Genome","pct"),("Reads Mapped Confidently to Genome","pct"),
 ("Reads Mapped Confidently to Intergenic Regions","pct"),("Reads Mapped Confidently to Intronic Regions","pct"),
 ("Reads Mapped Confidently to Exonic Regions","pct"),("Reads Mapped Confidently to Transcriptome","pct"),
 ("Reads Mapped Antisense to Gene","pct"),("Fraction Reads in Spots Under Tissue","pct")]
KIND = dict(SPEC)
a1_val = lambda l: num(a1raw[l]) if l in a1raw else None
d1_val = lambda l: (float(d1raw[l]) if l in d1raw else None)

def perspot(ad):
    ad.var_names_make_unique(); umi = np.asarray(ad.X.sum(1)).ravel(); genes = np.asarray((ad.X > 0).sum(1)).ravel()
    mt = ad.var_names.str.match(r"^mt-"); mito = np.asarray(ad[:, mt].X.sum(1)).ravel() / np.maximum(umi, 1)
    return dict(n=ad.n_obs, mean_genes=genes.mean(), mean_umi=umi.mean(),
                med_mito=np.median(mito), mean_mito=mito.mean(), umi_cv=umi.std()/umi.mean())
qa, qd = perspot(load_visium_a1()), perspot(load_visium("SMG:SLG_Outs/D1/outs", "D1", img_key="hires"))

wb = Workbook(); F = "Arial"
hf = PatternFill("solid", fgColor="1F4E79"); hft = Font(name=F, bold=True, size=12, color="FFFFFF")
cf = PatternFill("solid", fgColor="D9E1F2"); cft = Font(name=F, bold=True, size=11, color="1F4E79")
base = Font(name=F, size=11); border = Border(bottom=Side(style="thin", color="BFBFBF"))
CATS = [("Sequencing", ["Number of Reads","Valid Barcodes","Valid UMIs","Sequencing Saturation",
          "Q30 Bases in Barcode","Q30 Bases in RNA Read","Q30 Bases in UMI"]),
 ("Mapping", ["Reads Mapped to Genome","Reads Mapped Confidently to Genome","Reads Mapped Confidently to Exonic Regions",
          "Reads Mapped Confidently to Intronic Regions","Reads Mapped Confidently to Intergenic Regions",
          "Reads Mapped Confidently to Transcriptome","Reads Mapped Antisense to Gene"]),
 ("Spots & detection", ["Number of Spots Under Tissue","Fraction of Spots Under Tissue","Fraction Reads in Spots Under Tissue",
          "Mean Reads per Spot","Mean Reads Under Tissue per Spot","Median Genes per Spot","Median UMI Counts per Spot",
          "Total Genes Detected"])]
PS = [("Spots under tissue","n","#,##0"),("Mean genes per spot","mean_genes","#,##0"),("Mean UMIs per spot","mean_umi","#,##0"),
      ("Median mitochondrial fraction","med_mito","0.0%"),("Mean mitochondrial fraction","mean_mito","0.0%"),
      ("UMI coefficient of variation","umi_cv","0.00")]
nf = lambda k: "#,##0" if k == "int" else "0.0%"

def put(ws, r, lab, v, numfmt):
    ws.append([lab, v if v is not None else "n/a"]); ws[f"A{ws.max_row}"].font = base
    b = ws[f"B{ws.max_row}"]; b.font = base; b.alignment = Alignment(horizontal="right")
    if v is not None: b.number_format = numfmt
    for c in ws[ws.max_row]: c.border = border

def sheet(ws, gv, q, title, srver, sid):
    ws.append([title]); ws["A1"].font = Font(name=F, bold=True, size=13)
    ws.append([f"Sample: {sid}     Pipeline: {srver}"]); ws["A2"].font = Font(name=F, italic=True, size=10, color="595959")
    ws.append(["Metric", "Value"])
    for c in ws[3]: c.font = hft; c.fill = hf
    ws["B3"].alignment = Alignment(horizontal="right")
    for cat, labs in CATS:
        ws.append([cat, ""])
        for c in ws[ws.max_row]: c.font = cft; c.fill = cf
        for l in labs: put(ws, ws.max_row, l, gv(l), nf(KIND[l]))
    ws.append(["Per-spot QC (filtered matrix)", ""])
    for c in ws[ws.max_row]: c.font = cft; c.fill = cf
    for lab, key, numfmt in PS: put(ws, ws.max_row, lab, q[key], numfmt)
    ws.column_dimensions["A"].width = 44; ws.column_dimensions["B"].width = 16; ws.sheet_view.showGridLines = False

sheet(wb.active, a1_val, qa, "Visium QC — A1 (ORES foregut roll)", "spaceranger-1.2.1", "V10S15-395_A1")
wb.active.title = "A1 (ORES)"
sheet(wb.create_sheet("D1 (SMG-SLG)"), d1_val, qd, "Visium QC — D1 (submandibular / sublingual)", "spaceranger-3.0.0", "D1")

# comparison sheet
ws = wb.create_sheet("A1 vs D1")
ws.append(["Visium QC — A1 vs D1"]); ws["A1"].font = Font(name=F, bold=True, size=13)
ws.append(["Metric", "A1 (ORES)", "D1 (SMG/SLG)"])
for c in ws[2]: c.font = hft; c.fill = hf
ws["B2"].alignment = ws["C2"].alignment = Alignment(horizontal="right")
def put3(lab, a, d, numfmt):
    ws.append([lab, a if a is not None else "n/a", d if d is not None else "n/a"]); r = ws.max_row
    ws[f"A{r}"].font = base
    for col, val in (("B", a), ("C", d)):
        cc = ws[f"{col}{r}"]; cc.font = base; cc.alignment = Alignment(horizontal="right")
        if val is not None: cc.number_format = numfmt
    for c in ws[r]: c.border = border
for cat, labs in CATS:
    ws.append([cat, "", ""])
    for c in ws[ws.max_row]: c.font = cft; c.fill = cf
    for l in labs: put3(l, a1_val(l), d1_val(l), nf(KIND[l]))
ws.append(["Per-spot QC (filtered matrix)", "", ""])
for c in ws[ws.max_row]: c.font = cft; c.fill = cf
for lab, key, numfmt in PS: put3(lab, qa[key], qd[key], numfmt)
ws.column_dimensions["A"].width = 44; ws.column_dimensions["B"].width = 15; ws.column_dimensions["C"].width = 15
ws.sheet_view.showGridLines = False

out = "results/figures_parotid/Visium_QC_A1_D1.xlsx"; wb.save(out); print("wrote", out)
