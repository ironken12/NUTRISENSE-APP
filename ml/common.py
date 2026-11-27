"""Shared pieces for train.py and recalibrate.py: column lists, the thesis model scorer and the evaluation metrics.

Nothing here touches the app. The thesis model is scored with the app's own port (backend/app/ai/nutrisense_risk.py),
so a score here is exactly what the app would give.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
THESIS_MODEL = ROOT / "backend/app/ai/riskModel/ifls5-lr-1/nutrisense_risk_model.json"  # the thesis model (archived)

# The thesis model's 16 inputs, with the app's names (see riskModel/README.md).
NUMERIC = ["age_months", "birth_weight_g", "mother_height_cm", "antenatal_contacts", "mother_education"]
BINARY = ["sex_male", "low_birth_weight", "exclusive_bf_6m", "diarrhoea_1m", "ari_1m", "immunisation_complete",
          "mother_anaemia", "improved_water", "improved_sanitation", "urban"]
# Optional extra inputs a Kader can collect. Used when the column is in the data.
EXTRA_NUMERIC = ["maternal_age", "birth_order", "birth_interval_months", "father_height_cm", "n_under5", "wealth_index"]
EXTRA_BINARY = ["smoking_household"]
CONTEXT_PREFIX = "ctx_"  # district-level context (poverty rate, altitude, ...): works for districts the model never saw


def _load_port():
    spec = importlib.util.spec_from_file_location("nutrisense_risk", ROOT / "backend/app/ai/nutrisense_risk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_thesis_model(path: Path = THESIS_MODEL) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def thesis_p_high(df: pd.DataFrame, model: dict) -> np.ndarray:
    """P(high) from the thesis model, exactly as the app computes it."""
    port = _load_port()
    cols = NUMERIC + BINARY + ["district"]
    out = []
    for row in df[[c for c in cols if c in df.columns]].to_dict("records"):
        child = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items()}
        out.append(port.predict_risk(model, child)["proba"]["high"])
    return np.asarray(out)


ACTIVE_MODEL = ROOT / "backend/app/ai/riskModel/nutrisense_risk_model.json"  # ifls5-lr-2, the model the app runs


def active_p_stunting(df: pd.DataFrame, model: dict) -> np.ndarray:
    """P(stunting) from the active model (ifls5-lr-2), exactly as the app computes it."""
    port = _load_port()
    names = [model["age"]["name"]] + [f["name"] for f in model["numeric"]] + [f["name"] for f in model["binary"]]
    out = []
    for row in df.reindex(columns=names).to_dict("records"):
        child = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items()}
        out.append(port.predict_stunting(model, child)["p_stunting"])
    return np.asarray(out)


def make_target(df: pd.DataFrame, label_col: str | None) -> np.ndarray:
    """1 = high risk. Either the thesis label column (low/medium/high; high = 1) or stunting, HAZ < -2."""
    if label_col:
        return (df[label_col].astype(str).str.lower() == "high").to_numpy(int)
    return (df["haz"] < -2).to_numpy(int)


def weights(df: pd.DataFrame, weight_col: str | None) -> np.ndarray:
    if weight_col and weight_col in df.columns:
        w = df[weight_col].fillna(df[weight_col].median()).to_numpy(float)
        return w / w.mean()
    return np.ones(len(df))


# ---- metrics --------------------------------------------------------------------------------------------------------

def threshold_for_capacity(score: np.ndarray, w: np.ndarray, capacity: float) -> float:
    """The score above which the top `capacity` share (weighted) of children is flagged."""
    order = np.argsort(-score)
    cum = np.cumsum(w[order]) / w.sum()
    k = int(np.searchsorted(cum, capacity))
    return float(score[order][min(k, len(score) - 1)])


def threshold_for_recall(score: np.ndarray, y: np.ndarray, w: np.ndarray, recall: float) -> float:
    """The highest threshold that still catches `recall` of the positives."""
    pos = np.sort(score[y == 1])[::-1]
    wp = w[y == 1][np.argsort(-score[y == 1])]
    cum = np.cumsum(wp) / wp.sum()
    k = int(np.searchsorted(cum, recall))
    return float(pos[min(k, len(pos) - 1)])


def at_threshold(score, y, w, thr) -> dict:
    flag = score >= thr
    tp = w[flag & (y == 1)].sum()
    fl = w[flag].sum()
    pos = w[y == 1].sum()
    precision = tp / fl if fl else float("nan")
    return {"flagged": fl / w.sum(), "recall": tp / pos if pos else float("nan"), "precision": precision,
            "visits_per_case": 1 / precision if precision else float("nan")}


def calibration(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Calibration intercept and slope (logistic recalibration of logit(p) on y). Ideal: 0 and 1."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(penalty=None, max_iter=1000).fit(lp, y)
    return float(m.intercept_[0]), float(m.coef_[0][0])


def net_benefit(score, y, w, pts) -> list[float]:
    """Decision-curve net benefit at each threshold probability (only meaningful for calibrated probabilities)."""
    n = w.sum()
    out = []
    for pt in pts:
        flag = score >= pt
        tp = w[flag & (y == 1)].sum() / n
        fp = w[flag & (y == 0)].sum() / n
        out.append(tp - fp * pt / (1 - pt))
    return out


def evaluate(name, score, y, w, capacity, probabilistic=True) -> dict:
    thr = threshold_for_capacity(score, w, capacity)
    r = {"model": name, "auc": roc_auc_score(y, score, sample_weight=w), "threshold": thr,
         **{f"cap_{k}": v for k, v in at_threshold(score, y, w, thr).items()}}
    if probabilistic:
        r["brier"] = brier_score_loss(y, score, sample_weight=w)
        r["cal_intercept"], r["cal_slope"] = calibration(score, y)
    return r


def grouped_bootstrap(score, y, w, groups, capacity, reps=200, seed=42) -> dict:
    """95% CIs for AUC and recall at capacity, resampling households (children in one household are not independent)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    aucs, recalls = [], []
    for _ in range(reps):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx], sample_weight=w[idx]))
        thr = threshold_for_capacity(score[idx], w[idx], capacity)
        recalls.append(at_threshold(score[idx], y[idx], w[idx], thr)["recall"])
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return {"auc_ci": q(aucs), "recall_ci": q(recalls)}
