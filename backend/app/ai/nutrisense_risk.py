"""The thesis risk model (Section 4.7): a line-by-line Python port of riskModel/nutrisenseRisk.js.

The coefficients, threshold, imputation values, scaling and district columns come from riskModel/nutrisense_risk_model.json
(loaded once). The arithmetic follows the JavaScript exactly, in the same order, so the results match it and scikit-learn
to about 1e-16 (tests/test_nutrisense_risk.py runs the 300 parity vectors in riskModel/parity_test_vectors.json).
Do not change the maths here: change the JSON (a new model version) instead.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(__file__).with_name("riskModel")


@lru_cache(maxsize=1)
def load_model() -> dict:
    with open(MODEL_DIR / "nutrisense_risk_model.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_model_card() -> dict:
    with open(MODEL_DIR / "model_card.json", encoding="utf-8") as f:
        return json.load(f)


def _missing(v) -> bool:  # JS isMissing: null, undefined or NaN
    return v is None or (isinstance(v, float) and math.isnan(v))


def _js_string(v) -> str:  # JS String(v) for the district code: 5371 and 5371.0 both give "5371"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def feature_vector(model: dict, child: dict) -> list[float]:
    x: list[float] = []
    for f in model["numeric"]:
        v = f["impute"] if _missing(child.get(f["name"])) else float(child[f["name"]])
        x.append((v - f["mean"]) / f["scale"])
    for f in model["binary"]:
        x.append(f["impute"] if _missing(child.get(f["name"])) else float(child[f["name"]]))
    cat = model["categorical"]
    d = cat["missing_value"] if _missing(child.get(cat["name"])) else _js_string(child[cat["name"]])
    if d in cat["infrequent"]:
        d = "infrequent_sklearn"
    for c in cat["columns"]:
        x.append(1 if c == d else 0)  # unseen district (e.g. every NTT district) -> all zeros
    return x


def predict_risk(model: dict, child: dict) -> dict:
    """{proba: {low, medium, high}, level, modelVersion}, exactly as predictRisk() in nutrisenseRisk.js."""
    x = feature_vector(model, child)
    z = []
    for k, w in enumerate(model["coef"]):
        s = model["intercept"][k]
        for i, wi in enumerate(w):
            s = s + wi * x[i]
        z.append(s)
    m = max(z)
    e = [math.exp(v - m) for v in z]
    total = 0
    for v in e:
        total = total + v
    proba = [v / total for v in e]  # [low, medium, high]
    level = "high" if proba[2] >= model["threshold_high"] else ("low" if proba[0] >= proba[1] else "medium")
    return {"proba": {"low": proba[0], "medium": proba[1], "high": proba[2]}, "level": level, "modelVersion": model["version"]}
