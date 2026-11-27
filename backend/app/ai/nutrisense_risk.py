"""The risk model scorers: line-by-line Python ports of the JavaScript in riskModel/.

* ifls5-lr-2 (active, riskModel/): logistic regression with an age spline and missing-value indicators, P(stunting).
  predict_stunting() ports riskModel/nutrisenseRisk.js. Exported by ml/export_app_model.py.
* ifls5-lr-1 (thesis model, archived in riskModel/ifls5-lr-1/): multinomial logistic regression, P(low/medium/high).
  predict_risk() ports riskModel/ifls5-lr-1/nutrisenseRisk.js. Kept for reproducibility and comparison.

Both match scikit-learn to about 1e-16 (tests/test_nutrisense_risk.py runs the 300 parity vectors of each).
Do not change the maths here: change the JSON (a new model version) instead.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(__file__).with_name("riskModel")
THESIS_DIR = MODEL_DIR / "ifls5-lr-1"


@lru_cache(maxsize=1)
def load_model() -> dict:
    with open(MODEL_DIR / "nutrisense_risk_model.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_model_card() -> dict:
    with open(MODEL_DIR / "model_card.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_thesis_model() -> dict:
    """ifls5-lr-1, the model evaluated in the thesis (archived)."""
    with open(THESIS_DIR / "nutrisense_risk_model.json", encoding="utf-8") as f:
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


# ---- ifls5-lr-2: logistic regression with an age spline and missing-value indicators (ml/export_app_model.py) ----

def _bspline_basis(x: float, t: list[float], degree: int) -> list[float]:
    """Values of all B-spline basis functions of `degree` on knot vector `t` at x (Cox-de Boor), as
    scikit-learn's SplineTransformer computes them. x must lie within [t[degree], t[-degree-1]]."""
    n = len(t) - degree - 1
    # Knot span: t[k] <= x < t[k+1]; at the right end use the last non-empty span.
    k = degree
    while k < n - 1 and x >= t[k + 1]:
        k += 1
    # Degree-0 basis on span k, raised to `degree` (The NURBS Book, algorithm A2.2).
    N = [0.0] * (degree + 1)
    left = [0.0] * (degree + 1)
    right = [0.0] * (degree + 1)
    N[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = x - t[k + 1 - j]
        right[j] = t[k + j] - x
        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    out = [0.0] * n
    for i in range(degree + 1):
        out[k - degree + i] = N[i]
    return out


def feature_vector_v2(model: dict, child: dict) -> list[float]:
    age = model["age"]
    a = age["impute"] if _missing(child.get(age["name"])) else float(child[age["name"]])
    t, deg = age["knots"], age["degree"]
    a = min(max(a, t[deg]), t[-deg - 1])  # constant extrapolation
    x = _bspline_basis(a, t, deg)
    raw = [f["impute"] if _missing(child.get(f["name"])) else float(child[f["name"]]) for f in model["numeric"]]
    sc = model["scaling"]
    x += [(v - m) / s for v, m, s in zip(raw, sc["mean"], sc["scale"])]
    x += [f["impute"] if _missing(child.get(f["name"])) else float(child[f["name"]]) for f in model["binary"]]
    # Missing-value flags (0/1, not scaled); empty for the shipped model.
    x += [1.0 if _missing(child.get(n)) else 0.0 for n in model["numeric_missing_indicators"]]
    x += [1.0 if _missing(child.get(n)) else 0.0 for n in model["binary_missing_indicators"]]
    return x


def predict_stunting(model: dict, child: dict) -> dict:
    """{p_stunting, level, modelVersion} for an ifls5-lr-2 model. Unknown inputs: None (never 0)."""
    x = feature_vector_v2(model, child)
    z = model["intercept"]
    for w, v in zip(model["coef"], x):
        z = z + w * v
    p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    level = "high" if p >= model["threshold_high"] else ("medium" if p >= model["threshold_medium"] else "low")
    return {"p_stunting": p, "level": level, "modelVersion": model["version"]}
