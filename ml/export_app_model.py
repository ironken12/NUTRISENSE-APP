"""Train the final model (new logistic regression + extra inputs) and export it for the app.

    python ml/export_app_model.py --data children.csv --weight-col weight --holdout-province 52 \
        --out backend/app/ai/riskModel

Writes to --out:
  nutrisense_risk_model.json  coefficients, spline knots, imputation values, scaling and the two thresholds
  model_card.json             evaluation (random split and province hold-out), top inputs, caveats
  parity_test_vectors.json    300 SYNTHETIC children (no IFLS rows) scored by scikit-learn, for the app's parity tests

Levels: high = P(stunting) in the top --capacity-high share of the training children (default 30%, what Kader can
visit), medium = the next share up to --capacity-medium (default 60%), low = the rest. Thresholds are fixed on the
training data, so in a higher-risk area (NTT) more children are high.

The evaluation figures come from models trained without the test children (the same splits as train.py); the exported
model is then trained on all children.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import common as C
import train as T
from compare import paired

VERSION = "ifls5-lr-2"


def fit(df, args):
    cols = sum(T.feature_lists(df), [])
    pipe = T.logistic(df, df["hhid"].to_numpy(), args.seed).fit(
        df[cols], C.make_target(df, None), lr__sample_weight=C.weights(df, args.weight_col))
    return pipe, cols


def thresholds(p, w, args):
    return C.threshold_for_capacity(p, w, args.capacity_high), C.threshold_for_capacity(p, w, args.capacity_medium)


def evaluate(train, test, args):
    """Train on `train`, fix thresholds on `train`, judge on `test` (as deployed)."""
    pipe, cols = fit(train, args)
    t_high, _ = thresholds(pipe.predict_proba(train[cols])[:, 1], C.weights(train, args.weight_col), args)
    p = pipe.predict_proba(test[cols])[:, 1]
    y, w, g = C.make_target(test, None), C.weights(test, args.weight_col), test["hhid"].to_numpy()
    r = C.at_threshold(p, y, w, t_high)
    flag = p >= t_high
    acc = float(np.average(flag == (y == 1), weights=w))
    cal_int, cal_slope = C.calibration(p, y)
    rng = np.random.default_rng(args.seed)
    u = np.unique(g)
    ix = {k: np.flatnonzero(g == k) for k in u}
    aucs, recs = [], []
    for _ in range(args.bootstrap):
        i = np.concatenate([ix[k] for k in rng.choice(u, len(u))])
        aucs.append(roc_auc_score(y[i], p[i], sample_weight=w[i]))
        recs.append(C.at_threshold(p[i], y[i], w[i], t_high)["recall"])
    # Input importance: drop in AUC when the input is unknown for every test child (the model then imputes it).
    importance = {}
    base = roc_auc_score(y, p, sample_weight=w)
    # Paired comparisons on the same test children (household bootstrap): vs the thesis model and vs the simple rule.
    comparisons = {}
    for name, other in (("ifls5-lr-1", C.thesis_p_high(test, C.load_thesis_model())), ("rule", T.rule_score(test))):
        mu, lo, hi, share = paired(p, other, y, w, g, args.bootstrap, seed=7)
        comparisons[name] = {"auc_difference": round(float(mu), 3), "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
                             "share_not_better": round(float(share), 3)}
    for c in cols:
        t2 = test[cols].copy()
        t2[c] = np.nan
        importance[c] = base - roc_auc_score(y, pipe.predict_proba(t2)[:, 1], sample_weight=w)
    return {
        "n_test": int(len(test)), "prevalence_stunting": round(float(np.average(y, weights=w)), 3),
        "auc": round(base, 3), "auc_ci_low": round(float(np.percentile(aucs, 2.5)), 3),
        "auc_ci_high": round(float(np.percentile(aucs, 97.5)), 3),
        "flagged_high": round(r["flagged"], 3), "recall_high": round(r["recall"], 3),
        "recall_high_ci_low": round(float(np.percentile(recs, 2.5)), 3), "recall_high_ci_high": round(float(np.percentile(recs, 97.5)), 3),
        "precision_high": round(r["precision"], 3), "accuracy": round(acc, 3),
        "brier": round(float(np.average((p - y) ** 2, weights=w)), 3), "calibration_intercept": round(cal_int, 3),
        "slope": round(cal_slope, 3), "compared_with": comparisons,
    }, importance


def export_model(pipe, cols, t_high, t_med, n, args) -> dict:
    pre, lr = pipe.named_steps["pre"], pipe.named_steps["lr"]
    age_pipe = pre.named_transformers_["age"]
    num_pipe = pre.named_transformers_["num"]
    tr = {name: cols for name, _, cols in pre.transformers_}
    age_cols, num_cols, bin_cols = tr["age"], tr["num"], tr["bin"]
    spl = age_pipe.named_steps["spl"]
    num_imp, num_sc = num_pipe.named_steps["imp"], num_pipe.named_steps["sc"]
    bin_imp = pre.named_transformers_["bin"]
    flags = pre.named_transformers_
    num_ind = [num_cols[i] for i in flags["num_missing"].features_] if "num_missing" in flags else []
    bin_ind = [bin_cols[i] for i in flags["bin_missing"].features_] if "bin_missing" in flags else []
    # Column order of the ColumnTransformer: age spline, numeric (scaled), binary, numeric flags, binary flags.
    columns = ([f"age_spline_{i}" for i in range(spl.n_features_out_)] + num_cols + bin_cols
               + [f"{c}__missing" for c in num_ind] + [f"{c}__missing" for c in bin_ind])
    coef = lr.coef_[0].tolist()
    assert len(coef) == len(columns)
    return {
        "name": "NutriSense risk model", "version": VERSION, "kind": "logistic_spline_v2",
        "trained_on": f"IFLS5 (2014-15), {n} children aged 0-59 months, 13 provinces (no NTT); IFLS5 health-measurement weights",
        "target": "stunting, HAZ < -2 (WHO 2006)",
        "link": "logistic: P(stunting) = 1 / (1 + exp(-(intercept + coef . x)))",
        "threshold_high": t_high, "threshold_medium": t_med,
        "decision_rule": (f"high if P(stunting) >= threshold_high (top {args.capacity_high:.0%} of the training children), "
                          f"medium if >= threshold_medium (top {args.capacity_medium:.0%}), else low"),
        "age": {"name": age_cols[0], "impute": float(age_pipe.named_steps["imp"].statistics_[0]), "degree": int(spl.degree),
                "knots": spl.bsplines_[0].t.tolist(), "extrapolation": "constant (clip to the knot range)"},
        "numeric": [{"name": c, "impute": float(num_imp.statistics_[i])} for i, c in enumerate(num_cols)],
        "scaling": {"columns": num_cols, "mean": num_sc.mean_.tolist(), "scale": num_sc.scale_.tolist()},
        "numeric_missing_indicators": num_ind,  # 0/1 flags, not scaled (empty unless trained with --missing-flags)
        "binary": [{"name": c, "impute": float(bin_imp.statistics_[i])} for i, c in enumerate(bin_cols)],
        "binary_missing_indicators": bin_ind,
        "columns": columns, "coef": coef, "intercept": float(lr.intercept_[0]), "C": float(lr.C_[0]),
    }


def parity_vectors(pipe, cols, df, model, seed) -> dict:
    """300 synthetic children: values drawn independently within each input's observed range (no real rows), with
    about a fifth of values missing, plus edge cases. Scored by scikit-learn."""
    rng = np.random.default_rng(seed)
    n = 300
    data = {}
    for c in cols:
        s = df[c].dropna()
        lo, hi = float(s.min()), float(s.max())
        if c in C.BINARY + C.EXTRA_BINARY:
            v = rng.integers(0, 2, n).astype(float)
        elif c in ("mother_education", "birth_order", "n_under5", "wealth_index", "antenatal_contacts"):
            v = rng.integers(int(lo), int(hi) + 1, n).astype(float)
        else:
            v = np.round(rng.uniform(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), n), 2)  # a little outside: tests clipping
        v[rng.random(n) < 0.2] = np.nan
        data[c] = v
    X = pd.DataFrame(data)
    X.loc[0, :] = np.nan  # a child with no data
    X.loc[0, "age_months"] = np.nan
    X.loc[1, "age_months"] = 0.0
    X.loc[2, "age_months"] = 59.99
    p = pipe.predict_proba(X[cols])[:, 1]
    cases = []
    for i in range(n):
        child = {c: (None if pd.isna(X.loc[i, c]) else float(X.loc[i, c])) for c in cols}
        level = "high" if p[i] >= model["threshold_high"] else ("medium" if p[i] >= model["threshold_medium"] else "low")
        cases.append({"child": child, "p_stunting": float(p[i]), "level": level})
    return {"model_version": VERSION, "note": "SYNTHETIC children, not IFLS data; scored by scikit-learn", "cases": cases}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--weight-col", default=None)
    ap.add_argument("--holdout-province", default="52")
    ap.add_argument("--capacity-high", type=float, default=0.30)
    ap.add_argument("--capacity-medium", type=float, default=0.60)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--missing-flags", action="store_true", help="0/1 missing-value flags (see train.logistic)")
    args = ap.parse_args()
    T.MISSING_FLAGS = args.missing_flags
    df = pd.read_csv(args.data, dtype={"district": str, "province": str, "hhid": str})
    df = df[df["haz"].notna()].reset_index(drop=True)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(df, groups=df["hhid"]))
    ev_rand, imp = evaluate(df.iloc[tr], df.iloc[te], args)
    m = df["province"] == str(args.holdout_province)
    ev_hold, _ = evaluate(df[~m], df[m], args)

    pipe, cols = fit(df, args)
    p = pipe.predict_proba(df[cols])[:, 1]
    t_high, t_med = thresholds(p, C.weights(df, args.weight_col), args)
    model = export_model(pipe, cols, t_high, t_med, len(df), args)
    top = [c for c, _ in sorted(imp.items(), key=lambda t: -t[1])[:5]]
    card = {
        "model": "Logistic regression with age spline (improved model)",
        "data": f"IFLS5 2014-15, {len(df):,} children aged 0-59 months with a measured height, 13 provinces (no NTT)",
        "target": "Stunting (HAZ < -2, WHO 2006)",
        "excludes": "child height, weight and z-scores, district, regional prevalence and 2T",
        "decision_rule": model["decision_rule"],
        "evaluation": {"Random 80/20 split": ev_rand, f"NTB hold-out": ev_hold},
        "top_inputs": top,
        "input_importance_auc_drop": {k: round(v, 4) for k, v in sorted(imp.items(), key=lambda t: -t[1])},
        "comparisons_note": ("AUC difference of this model minus ifls5-lr-1 (the thesis model) and minus the rule 'mother < 150 cm "
                             "or birth weight < 2.5 kg', on the same test children (paired household bootstrap). ifls5-lr-1 was "
                             "trained on the random split of IFLS5, so it may have seen some test children: its side is optimistic."),
        "caveats": [
            "Not a clinical validation",
            "Modest discrimination: for prioritising Kader visits, not for diagnosis",
            "Under-predicts risk where stunting is more common (NTB); recalibrate for NTT",
            "Trained on 2014-15 data without NTT; immunisation known for only 22% of children (vaccination card)",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "nutrisense_risk_model.json").write_text(json.dumps(model, indent=1))
    (args.out / "model_card.json").write_text(json.dumps(card, indent=1))
    (args.out / "parity_test_vectors.json").write_text(json.dumps(parity_vectors(pipe, cols, df, model, args.seed), indent=1))
    print(json.dumps({"thresholds": [t_high, t_med], "evaluation": card["evaluation"], "top_inputs": top}, indent=1))


if __name__ == "__main__":
    main()
