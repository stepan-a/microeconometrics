"""
Application: Default of Credit Card Clients (Taiwan, 2005)
Estimation Logit / Probit + interprétation des résultats.

Source: UCI ML Repository
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

# ── 1. Téléchargement et préparation des données ─────────────────────────

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(DATA_DIR, "credit_default.zip")
XLS_PATH = os.path.join(DATA_DIR, "default of credit card clients.xls")

if not os.path.exists(XLS_PATH):
    url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    print("Téléchargement des données...")
    urllib.request.urlretrieve(url, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(ZIP_PATH)

df = pd.read_excel(XLS_PATH, header=1)
df.columns = df.columns.str.strip()

# Renommage pour lisibilité
df = df.rename(columns={
    "LIMIT_BAL": "LIMIT",
    "PAY_0": "PAY_SEPT",
    "default payment next month": "DEFAULT",
})

# Variables binaires
df["FEMALE"] = (df["SEX"] == 2).astype(int)
df["MARRIED"] = (df["MARRIAGE"] == 1).astype(int)
df["UNIVERSITY"] = (df["EDUCATION"] == 2).astype(int)
df["GRADUATE"] = (df["EDUCATION"] == 1).astype(int)
# Retard de paiement en septembre (>=1 mois de retard)
df["LATE_SEPT"] = (df["PAY_SEPT"] >= 1).astype(int)
# Limite de crédit en milliers de NT$
df["LIMIT_K"] = df["LIMIT"] / 1000

# ── 2. Variables du modèle ───────────────────────────────────────────────

y = df["DEFAULT"]
X_vars = ["LIMIT_K", "FEMALE", "MARRIED", "GRADUATE", "UNIVERSITY",
          "AGE", "LATE_SEPT"]
X = sm.add_constant(df[X_vars])

print(f"\nN = {len(y)}, taux de défaut = {y.mean():.3f}")
print(f"\nStatistiques descriptives :")
print(df[X_vars + ["DEFAULT"]].describe().round(2))

# ── 3. Estimation Logit et Probit ────────────────────────────────────────

logit = sm.Logit(y, X).fit(disp=0)
probit = sm.Probit(y, X).fit(disp=0)

print("\n" + "=" * 70)
print("RÉSULTATS LOGIT")
print("=" * 70)
print(logit.summary())

print("\n" + "=" * 70)
print("RÉSULTATS PROBIT")
print("=" * 70)
print(probit.summary())

# ── 4. Tableau comparatif pour LaTeX ─────────────────────────────────────

def significance(pval):
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return ""

var_labels = {
    "const": "Constante",
    "LIMIT_K": "Limite crédit (milliers NT\\$)",
    "FEMALE": "Femme",
    "MARRIED": "Marié(e)",
    "GRADUATE": "Diplôme 3\\textsuperscript{e} cycle",
    "UNIVERSITY": "Diplôme universitaire",
    "AGE": "Âge",
    "LATE_SEPT": "Retard paiement sept.",
}

print("\n\n% ── Tableau LaTeX : Logit vs Probit ──")
print("\\begin{tabular}{lrrrr}")
print("    \\toprule")
print("    & \\multicolumn{2}{c}{\\textbf{Logit}} & \\multicolumn{2}{c}{\\textbf{Probit}} \\\\")
print("    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
print("    \\textbf{Variable} & \\textbf{Coef.} & \\textbf{(é.t.)} & \\textbf{Coef.} & \\textbf{(é.t.)} \\\\")
print("    \\midrule")
for var in X.columns:
    label = var_labels.get(var, var)
    lc = logit.params[var]
    ls = logit.bse[var]
    lsig = significance(logit.pvalues[var])
    pc = probit.params[var]
    ps = probit.bse[var]
    psig = significance(probit.pvalues[var])
    print(f"    {label} & ${lc:+.4f}${lsig} & $({ls:.4f})$ & ${pc:+.4f}${psig} & $({ps:.4f})$ \\\\")
print("    \\midrule")
print(f"    Log-vraisemblance & \\multicolumn{{2}}{{c}}{{${logit.llf:.1f}$}} & \\multicolumn{{2}}{{c}}{{${probit.llf:.1f}$}} \\\\")
print(f"    AIC & \\multicolumn{{2}}{{c}}{{${logit.aic:.1f}$}} & \\multicolumn{{2}}{{c}}{{${probit.aic:.1f}$}} \\\\")
print(f"    $N$ & \\multicolumn{{2}}{{c}}{{${len(y)}$}} & \\multicolumn{{2}}{{c}}{{${len(y)}$}} \\\\")
print("    \\bottomrule")
print("\\end{tabular}")

# ── 5. Conversion Logit → Probit ─────────────────────────────────────────

ratio = np.sqrt(3) / np.pi
print("\n\n% ── Vérification conversion ──")
print(f"% Facteur théorique : sqrt(3)/pi = {ratio:.4f}")
for var in X.columns:
    if var == "const":
        continue
    converted = logit.params[var] * ratio
    actual = probit.params[var]
    print(f"% {var:15s}: Logit*0.55 = {converted:+.4f}, Probit = {actual:+.4f}")

# ── 6. Odds ratios (Logit) ──────────────────────────────────────────────

print("\n\n% ── Odds ratios (Logit) ──")
print("\\begin{tabular}{lrrr}")
print("    \\toprule")
print("    \\textbf{Variable} & $\\hat{\\beta}$ & $\\exp(\\hat{\\beta})$ & \\textbf{Interprétation} \\\\")
print("    \\midrule")
for var in X_vars:
    label = var_labels.get(var, var)
    coef = logit.params[var]
    odds = np.exp(coef)
    pct = (odds - 1) * 100
    print(f"    {label} & ${coef:+.4f}$ & ${odds:.4f}$ & ${pct:+.1f}$\\% \\\\")
print("    \\bottomrule")
print("\\end{tabular}")

# ── 7. Effets marginaux au point moyen ───────────────────────────────────

mfx_logit = logit.get_margeff(at="mean")
mfx_probit = probit.get_margeff(at="mean")

print("\n\n% ── Effets marginaux au point moyen ──")
print("\\begin{tabular}{lrr}")
print("    \\toprule")
print("    \\textbf{Variable} & \\textbf{Logit} & \\textbf{Probit} \\\\")
print("    \\midrule")
for i, var in enumerate(X_vars):
    label = var_labels.get(var, var)
    ml = mfx_logit.margeff[i]
    mp = mfx_probit.margeff[i]
    sl = significance(mfx_logit.pvalues[i])
    sp = significance(mfx_probit.pvalues[i])
    print(f"    {label} & ${ml:+.4f}${sl} & ${mp:+.4f}${sp} \\\\")
print("    \\bottomrule")
print("\\end{tabular}")

# ── 8. Effets incrémentaux (variables binaires) ─────────────────────────

print("\n\n% ── Effets incrémentaux (Logit) ──")
binary_vars = ["FEMALE", "MARRIED", "GRADUATE", "UNIVERSITY", "LATE_SEPT"]
X_mean = X.mean()
m0 = logit.predict(X_mean.to_frame().T).values[0]

for var in binary_vars:
    X1 = X_mean.copy()
    X0 = X_mean.copy()
    X1[var] = 1
    X0[var] = 0
    p1 = logit.predict(X1.to_frame().T).values[0]
    p0 = logit.predict(X0.to_frame().T).values[0]
    delta = p1 - p0
    label = var_labels.get(var, var)
    print(f"% {label:35s}: P(Y=1|X=1) = {p1:.4f}, P(Y=1|X=0) = {p0:.4f}, Delta = {delta:+.4f}")

# ── 9. Effet interdécile (variables quantitatives) ──────────────────────

print("\n\n% ── Effets interdéciles (Logit) ──")
quant_vars = ["LIMIT_K", "AGE"]
for var in quant_vars:
    col = "LIMIT" if var == "LIMIT_K" else var
    d1 = df[col].quantile(0.1)
    d9 = df[col].quantile(0.9)
    if var == "LIMIT_K":
        d1 = d1 / 1000
        d9 = d9 / 1000
    X_d1 = X_mean.copy()
    X_d9 = X_mean.copy()
    X_d1[var] = d1
    X_d9[var] = d9
    p_d1 = logit.predict(X_d1.to_frame().T).values[0]
    p_d9 = logit.predict(X_d9.to_frame().T).values[0]
    delta = p_d9 - p_d1
    label = var_labels.get(var, var)
    print(f"% {label:35s}: D1={d1:.1f}, D9={d9:.1f}, P(D1)={p_d1:.4f}, P(D9)={p_d9:.4f}, Delta={delta:+.4f}")
