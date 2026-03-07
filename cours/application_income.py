"""
Application : Adult Income (US Census, 1994)
Estimation Logit / Probit + interprétation des résultats.

Source: UCI ML Repository
https://archive.ics.uci.edu/dataset/2/adult
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

# ── 1. Téléchargement et préparation des données ─────────────────────────

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "adult.data")

if not os.path.exists(CSV_PATH):
    import urllib.request
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
        "https://people.sc.fsu.edu/~jburkardt/datasets/census_income/adult.data",
    ]
    for url in urls:
        try:
            print(f"Téléchargement depuis {url}...")
            urllib.request.urlretrieve(url, CSV_PATH)
            break
        except Exception as e:
            print(f"  Échec : {e}")
    else:
        raise RuntimeError("Impossible de télécharger les données.")

col_names = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

df = pd.read_csv(CSV_PATH, header=None, names=col_names,
                  skipinitialspace=True, na_values="?")
df = df.dropna()

# Variable expliquée
df["HIGH_INCOME"] = (df["income"] == ">50K").astype(int)

# Variables explicatives
df["FEMALE"] = (df["sex"] == "Female").astype(int)
df["MARRIED"] = df["marital_status"].isin(
    ["Married-civ-spouse", "Married-AF-spouse"]).astype(int)
df["WHITE"] = (df["race"] == "White").astype(int)
df["BACHELOR"] = (df["education_num"] >= 13).astype(int)  # Bachelor+
df["HIGH_SCHOOL"] = ((df["education_num"] >= 9) &
                      (df["education_num"] <= 12)).astype(int)

# ── 2. Variables du modèle ───────────────────────────────────────────────

y = df["HIGH_INCOME"]
X_vars = ["age", "FEMALE", "MARRIED", "WHITE", "BACHELOR",
          "HIGH_SCHOOL", "hours_per_week"]
X = sm.add_constant(df[X_vars])

print(f"N = {len(y)}, taux revenu élevé = {y.mean():.3f}")
print(f"\nStatistiques descriptives :")
print(df[X_vars + ["HIGH_INCOME"]].describe().round(2))

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
    "age": "Âge",
    "FEMALE": "Femme",
    "MARRIED": "Marié(e)",
    "WHITE": "Blanc",
    "BACHELOR": "Diplôme $\\geq$ Bachelor",
    "HIGH_SCHOOL": "High school / College",
    "hours_per_week": "Heures / semaine",
}

print("\n\n% ── Tableau LaTeX : Logit vs Probit ──")
for var in X.columns:
    label = var_labels.get(var, var)
    lc = logit.params[var]
    ls = logit.bse[var]
    lsig = significance(logit.pvalues[var])
    pc = probit.params[var]
    ps = probit.bse[var]
    psig = significance(probit.pvalues[var])
    print(f"    {label} & ${lc:+.4f}${lsig} & $({ls:.4f})$ & ${pc:+.4f}${psig} & $({ps:.4f})$ \\\\")
print(f"    Log-vraisemblance & \\multicolumn{{2}}{{c}}{{${logit.llf:.1f}$}} & \\multicolumn{{2}}{{c}}{{${probit.llf:.1f}$}} \\\\")
print(f"    AIC & \\multicolumn{{2}}{{c}}{{${logit.aic:.1f}$}} & \\multicolumn{{2}}{{c}}{{${probit.aic:.1f}$}} \\\\")
print(f"    $N$ & \\multicolumn{{2}}{{c}}{{${len(y)}$}} & \\multicolumn{{2}}{{c}}{{${len(y)}$}} \\\\")

# ── 5. Conversion Logit → Probit ─────────────────────────────────────────

ratio = np.sqrt(3) / np.pi
print("\n\n% ── Vérification conversion ──")
print(f"% Facteur théorique : sqrt(3)/pi = {ratio:.4f}")
for var in X.columns:
    if var == "const":
        continue
    converted = logit.params[var] * ratio
    actual = probit.params[var]
    print(f"% {var:20s}: Logit*0.55 = {converted:+.4f}, Probit = {actual:+.4f}")

# ── 6. Odds ratios (Logit) ──────────────────────────────────────────────

print("\n\n% ── Odds ratios (Logit) ──")
for var in X_vars:
    label = var_labels.get(var, var)
    coef = logit.params[var]
    odds = np.exp(coef)
    pct = (odds - 1) * 100
    print(f"    {label} & ${coef:+.4f}$ & ${odds:.4f}$ & ${pct:+.1f}$\\% \\\\")

# ── 7. Effets marginaux au point moyen ───────────────────────────────────

mfx_logit = logit.get_margeff(at="mean")
mfx_probit = probit.get_margeff(at="mean")

print("\n\n% ── Effets marginaux au point moyen ──")
for i, var in enumerate(X_vars):
    label = var_labels.get(var, var)
    ml = mfx_logit.margeff[i]
    mp = mfx_probit.margeff[i]
    sl = significance(mfx_logit.pvalues[i])
    sp = significance(mfx_probit.pvalues[i])
    print(f"    {label} & ${ml:+.4f}${sl} & ${mp:+.4f}${sp} \\\\")

# ── 8. Effets incrémentaux (variables binaires) ─────────────────────────

print("\n\n% ── Effets incrémentaux (Logit) ──")
binary_vars = ["FEMALE", "MARRIED", "WHITE", "BACHELOR", "HIGH_SCHOOL"]
X_mean = X.mean()

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
quant_vars = ["age", "hours_per_week"]
for var in quant_vars:
    d1 = df[var].quantile(0.1)
    d9 = df[var].quantile(0.9)
    X_d1 = X_mean.copy()
    X_d9 = X_mean.copy()
    X_d1[var] = d1
    X_d9[var] = d9
    p_d1 = logit.predict(X_d1.to_frame().T).values[0]
    p_d9 = logit.predict(X_d9.to_frame().T).values[0]
    delta = p_d9 - p_d1
    label = var_labels.get(var, var)
    print(f"% {label:35s}: D1={d1:.0f}, D9={d9:.0f}, P(D1)={p_d1:.4f}, P(D9)={p_d9:.4f}, Delta={delta:+.4f}")

# ── 10. Stats descriptives pour les slides ──────────────────────────────

print("\n\n% ── Stats descriptives ──")
print("% Variables binaires :")
for v in binary_vars:
    r0 = df.loc[df[v] == 0, "HIGH_INCOME"].mean()
    r1 = df.loc[df[v] == 1, "HIGH_INCOME"].mean()
    print(f"% {v:20s}: prop={df[v].mean():.3f}, taux(0)={r0:.3f}, taux(1)={r1:.3f}")

print("% Variables quantitatives :")
for v in quant_vars:
    print(f"% {v:20s}: mean={df[v].mean():.1f}, std={df[v].std():.1f}, "
          f"min={df[v].min():.0f}, max={df[v].max():.0f}, med={df[v].median():.0f}")

print("\n% Taux revenu élevé par tranche d'âge :")
bins = [17, 25, 35, 45, 55, 65, 100]
labels = ["17-24", "25-34", "35-44", "45-54", "55-64", "65+"]
df["AGE_BIN"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
print(df.groupby("AGE_BIN", observed=False)["HIGH_INCOME"].agg(["mean", "count"]))
