#!/usr/bin/env python3
"""
Calcul du MSE de l'estimateur du maximum de vraisemblance
pour le paramètre p de la loi géométrique.

L'estimateur est p̂ = n/S où S = Σᵢ Xᵢ et Xᵢ ~ Géom(p).
S suit une loi binomiale négative NegBin(n, p).
"""

import numpy as np
from scipy import integrate


def E_inv_S(n: int, p: float) -> float:
    """
    Calcule E[1/S] où S ~ NegBin(n, p).

    Utilise la représentation intégrale:
    E[1/S] = p^n ∫₀¹ t^(n-1) / (1-qt)^n dt
    """
    q = 1 - p

    def integrand(t):
        if t == 0:
            return 0
        return t**(n-1) / (1 - q*t)**n

    result, _ = integrate.quad(integrand, 0, 1)
    return p**n * result


def E_inv_S2(n: int, p: float) -> float:
    """
    Calcule E[1/S²] où S ~ NegBin(n, p).

    Utilise la représentation intégrale:
    E[1/S²] = p^n ∫₀¹ (-ln t) t^(n-1) / (1-qt)^n dt
    """
    q = 1 - p

    def integrand(t):
        if t <= 0:
            return 0
        return (-np.log(t)) * t**(n-1) / (1 - q*t)**n

    result, _ = integrate.quad(integrand, 1e-15, 1)
    return p**n * result


def statistiques_emv(n: int, p: float) -> dict:
    """
    Calcule les statistiques de l'EMV p̂ = n/S.

    Retourne un dictionnaire avec:
    - E_p_hat: espérance de l'estimateur
    - variance: variance de l'estimateur
    - biais: biais de l'estimateur
    - mse: erreur quadratique moyenne
    - cramer_rao: borne de Cramér-Rao
    """
    e1 = E_inv_S(n, p)
    e2 = E_inv_S2(n, p)

    E_p_hat = n * e1
    variance = n**2 * e2 - n**2 * e1**2
    biais = E_p_hat - p
    mse = variance + biais**2
    cramer_rao = p**2 * (1 - p) / n

    return {
        'E_p_hat': E_p_hat,
        'variance': variance,
        'biais': biais,
        'mse': mse,
        'cramer_rao': cramer_rao,
        'ratio_cr_variance': cramer_rao / variance
    }


def tableau_mse(valeurs_n: list, valeurs_p: list) -> None:
    """Affiche un tableau comparatif MSE vs borne CR."""

    print("=" * 90)
    print(f"{'n':>4} | {'p':>5} | {'E[p̂]':>8} | {'Biais':>9} | "
          f"{'Variance':>11} | {'MSE':>11} | {'CR':>11} | {'CR/MSE':>7}")
    print("=" * 90)

    for n in valeurs_n:
        for p in valeurs_p:
            s = statistiques_emv(n, p)
            print(f"{n:>4} | {p:>5.2f} | {s['E_p_hat']:>8.4f} | "
                  f"{s['biais']:>+9.5f} | {s['variance']:>11.6f} | "
                  f"{s['mse']:>11.6f} | {s['cramer_rao']:>11.6f} | "
                  f"{s['ratio_cr_variance']:>6.1%}")
        print("-" * 90)


def tableau_latex(valeurs_n: list, p: float) -> str:
    """Génère un tableau LaTeX pour une valeur de p donnée."""

    lignes = []
    lignes.append(r"\begin{tabular}{ccccc}")
    lignes.append(r"\hline")
    lignes.append(r"$n$ & Biais$^2$ & Variance & MSE & CR/Variance \\")
    lignes.append(r"\hline")

    for n in valeurs_n:
        s = statistiques_emv(n, p)
        biais2 = s['biais']**2
        lignes.append(f"{n} & {biais2:.3f} & {s['variance']:.3f} & "
                      f"{s['mse']:.3f} & {s['ratio_cr_variance']:.0%} \\\\")

    lignes.append(r"\hline")
    lignes.append(r"\end{tabular}")

    return "\n".join(lignes)


if __name__ == "__main__":
    # Valeurs à tester
    valeurs_n = [1, 2, 5, 10, 20, 50, 100]
    valeurs_p = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print("\n" + "=" * 50)
    print("TABLEAU COMPLET : Variance vs Borne de Cramér-Rao")
    print("=" * 50 + "\n")

    tableau_mse(valeurs_n, valeurs_p)

    # Tableaux LaTeX pour différentes valeurs de p
    print("\n\n" + "=" * 50)
    print("TABLEAUX LATEX")
    print("=" * 50)

    for p in [0.3, 0.5, 0.7]:
        print(f"\n% Tableau pour p = {p}")
        print(tableau_latex([1, 5, 10, 50], p))
