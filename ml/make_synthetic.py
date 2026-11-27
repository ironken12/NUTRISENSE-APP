"""SYNTHETIC test data for train.py and recalibrate.py. Not IFLS, not NTT, not real children.

    python ml/make_synthetic.py --n 4000 --out /tmp/synthetic.csv
    python ml/make_synthetic.py --n 600 --ntt --out /tmp/synthetic_ntt.csv

It only checks that the scripts run end to end. Any figure computed from it means nothing: never report one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def make(n: int, ntt: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if ntt:
        districts = ["5301", "5302", "5303", "5304", "5314", "5371"]
    else:
        cols = json.loads((ROOT / "backend/app/ai/riskModel/nutrisense_risk_model.json").read_text())["categorical"]["columns"]
        districts = [c for c in cols if c.isdigit()][:30] + ["5201", "5202", "5271"]  # incl. NTB (52)
    hh = rng.integers(0, n // 2, n)
    d = rng.choice(districts, n)
    mh = rng.normal(150 if ntt else 151.7, 5.5, n)
    bw = rng.normal(2950 if ntt else 3130, 480, n)
    edu = rng.integers(0, 5, n).astype(float)
    age = rng.uniform(0, 59, n)
    urban = (rng.random(n) < (0.2 if ntt else 0.5)).astype(float)
    imm = (rng.random(n) < 0.7).astype(float)
    san = (rng.random(n) < (0.5 if ntt else 0.7)).astype(float)
    lin = (-0.9 + 0.08 * (151 - mh) + 0.0012 * (3100 - bw) - 0.25 * edu + 0.025 * np.minimum(age, 24) - 0.4 * urban
           - 0.3 * imm - 0.3 * san + (0.6 if ntt else 0.0) + rng.normal(0, 0.8, n))
    haz = -2.5 - 0.6 * lin + rng.normal(0, 0.6, n)  # about a third stunted, as in IFLS5
    df = pd.DataFrame({
        "hhid": [f"h{x}" for x in hh], "district": d, "province": [x[:2] for x in d], "age_months": age,
        "sex_male": (rng.random(n) < 0.5).astype(float), "birth_weight_g": bw.round(), "low_birth_weight": (bw < 2500).astype(float),
        "mother_height_cm": mh.round(1), "antenatal_contacts": rng.integers(0, 12, n).astype(float), "mother_education": edu,
        "exclusive_bf_6m": np.where(age < 6, np.nan, (rng.random(n) < 0.5).astype(float)),
        "diarrhoea_1m": (rng.random(n) < 0.12).astype(float), "ari_1m": (rng.random(n) < 0.08).astype(float),
        "immunisation_complete": np.where(age < 12, np.nan, imm), "mother_anaemia": (rng.random(n) < 0.3).astype(float),
        "improved_water": (rng.random(n) < 0.8).astype(float), "improved_sanitation": san, "urban": urban,
        "maternal_age": rng.normal(28, 6, n).round(), "n_under5": rng.integers(1, 4, n).astype(float),
        "weight": rng.uniform(0.5, 2.0, n), "haz": haz.round(2),
    })
    for col, share in (("mother_height_cm", 0.25), ("antenatal_contacts", 0.3), ("birth_weight_g", 0.15), ("mother_anaemia", 0.4)):
        df.loc[rng.random(n) < share, col] = np.nan  # blanks, as in the field
    df.loc[df["birth_weight_g"].isna(), "low_birth_weight"] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--ntt", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    make(a.n, a.ntt, a.seed).to_csv(a.out, index=False)
    print(f"wrote SYNTHETIC data: {a.out}")
