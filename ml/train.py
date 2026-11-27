"""Retrain and evaluate the NutriSense risk model with the improvements from the model review.

    python ml/train.py --data ifls5_children.csv --holdout-province 52 --out ml/out

What it does, compared with the thesis model (ifls5-lr-1):
  * binary target (high risk vs not) instead of three unordered levels;
  * no district dummy columns (every NTT district is unseen); district context columns (ctx_*) instead, if given;
  * age as a spline (risk rises steeply from 6 to 24 months);
  * regularisation chosen by cross-validation on households;
  * missing-value indicators, so blank fields (common in the field) are learned, not filled with the mean;
  * survey weights, if given;
  * threshold chosen by Kader capacity (share of children that can be visited), not a fixed probability.

It evaluates on a household-grouped random 80/20 split and, if --holdout-province is set, on that province held out.
Each setting compares: the thesis model (scored exactly as the app does), the new logistic regression, gradient boosting
and the simple rule "mother < 150 cm or birth weight < 2.5 kg". Writes report.md, figures and model.joblib to --out.

Input CSV: one row per child. Required: the thesis inputs (app names: see common.py), `hhid` (household id),
`province` (BPS code), and either `haz` (target = HAZ < -2) or --label-col (thesis labels low/medium/high).
Optional: --weight-col, the extra inputs in common.EXTRA_*, and ctx_* district context columns. Child height, weight
and z-scores must not be inputs (haz is only the target).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import SplineTransformer, StandardScaler  # noqa: E402

import common as C  # noqa: E402

FORBIDDEN = {"haz", "waz", "whz", "height_cm", "weight_kg", "child_height_cm", "child_weight_kg", "two_t", "prevalence"}


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = [c for c in C.NUMERIC + C.EXTRA_NUMERIC if c in df.columns and c != "age_months"]
    numeric += [c for c in df.columns if c.startswith(C.CONTEXT_PREFIX)]
    binary = [c for c in C.BINARY + C.EXTRA_BINARY if c in df.columns]
    leaked = FORBIDDEN & set(numeric + binary)
    if leaked:
        raise SystemExit(f"These columns must not be model inputs: {sorted(leaked)}")
    return ["age_months"], numeric, binary


def logistic(df: pd.DataFrame, groups: np.ndarray, seed: int) -> Pipeline:
    age, numeric, binary = feature_lists(df)
    pre = ColumnTransformer([
        ("age", Pipeline([("imp", SimpleImputer(strategy="median")), ("spl", SplineTransformer(n_knots=5, degree=3))]), age),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), numeric),
        ("bin", SimpleImputer(strategy="most_frequent", add_indicator=True), binary),
    ])
    folds = list(GroupKFold(n_splits=5).split(df, groups=groups))
    lr = LogisticRegressionCV(Cs=np.logspace(-3, 2, 12), cv=folds, scoring="neg_log_loss", max_iter=5000)
    return Pipeline([("pre", pre), ("lr", lr)])


def boosting(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=True, random_state=seed)


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Mother < 150 cm or birth weight < 2.5 kg (unknown counts as no): 0, 1 or 2 risk factors."""
    return ((df["mother_height_cm"] < 150).astype(int) + (df["birth_weight_g"] < 2500).astype(int)).to_numpy(float)


def run_setting(name, train, test, args, thesis_model) -> tuple[list[dict], dict, dict]:
    ytr, yte = C.make_target(train, args.label_col), C.make_target(test, args.label_col)
    wtr, wte = C.weights(train, args.weight_col), C.weights(test, args.weight_col)
    gtr, gte = train["hhid"].to_numpy(), test["hhid"].to_numpy()
    cols = sum(feature_lists(train), [])

    lr = logistic(train, gtr, args.seed).fit(train[cols], ytr, lr__sample_weight=wtr)
    gb = boosting(args.seed).fit(train[cols], ytr, sample_weight=wtr)
    scores = {
        "Thesis model (ifls5-lr-1), P(high)": (C.thesis_p_high(test, thesis_model), True),
        "New logistic regression": (lr.predict_proba(test[cols])[:, 1], True),
        "Gradient boosting": (gb.predict_proba(test[cols])[:, 1], True),
        "Rule: mother <150 cm or BW <2.5 kg": (rule_score(test), False),
    }
    rows, cis = [], {}
    for m, (s, prob) in scores.items():
        r = C.evaluate(m, s, yte, wte, args.capacity, probabilistic=prob)
        r["setting"], r["n_test"], r["prevalence"] = name, len(test), float(np.average(yte, weights=wte))
        rows.append(r)
        cis[m] = C.grouped_bootstrap(s, yte, wte, gte, args.capacity, reps=args.bootstrap, seed=args.seed)
    figures(name, scores, yte, wte, args.out)
    return rows, cis, {"lr": lr, "gb": gb, "C": float(lr.named_steps["lr"].C_[0])}


def figures(name, scores, y, w, out: Path):
    slug = name.lower().replace(" ", "-").replace("/", "-")
    pts = np.linspace(0.05, 0.6, 40)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    prev = np.average(y, weights=w)
    ax[0].plot(pts, prev - (1 - prev) * pts / (1 - pts), "--", color="grey", label="Visit everyone")
    ax[0].axhline(0, color="black", lw=0.8, label="Visit no one")
    for m, (s, prob) in scores.items():
        if prob:
            ax[0].plot(pts, C.net_benefit(s, y, w, pts), label=m)
            bins = np.quantile(s, np.linspace(0, 1, 11))
            idx = np.clip(np.digitize(s, bins[1:-1]), 0, 9)
            mp = [np.average(s[idx == b], weights=w[idx == b]) for b in range(10) if (idx == b).any()]
            mo = [np.average(y[idx == b], weights=w[idx == b]) for b in range(10) if (idx == b).any()]
            ax[1].plot(mp, mo, "o-", label=m)
    ax[0].set(title=f"Decision curve: {name}", xlabel="Threshold probability", ylabel="Net benefit",
              ylim=(-0.05, max(prev, 0.05) * 1.2))
    ax[1].plot([0, 1], [0, 1], ":", color="grey")
    ax[1].set(title=f"Calibration (deciles): {name}", xlabel="Predicted", ylabel="Observed")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"curves-{slug}.png", dpi=150)
    plt.close(fig)


def report(rows, cis, info, args) -> str:
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    lines = ["# Risk model retraining report", "",
             f"Data: `{args.data}` · target: {'thesis label = high' if args.label_col else 'stunting (HAZ < -2)'} · "
             f"capacity: Kader visit the top {pct(args.capacity)} · bootstrap: {args.bootstrap} household resamples", ""]
    for setting in dict.fromkeys(r["setting"] for r in rows):
        rs = [r for r in rows if r["setting"] == setting]
        lines += [f"## {setting}", "", f"Test children: {rs[0]['n_test']} · prevalence of the target: {pct(rs[0]['prevalence'])}"
                  f" · new LR regularisation C = {info[setting]['C']:.3g}", "",
                  "| Model | AUC (95% CI) | Recall at capacity (95% CI) | Precision | Visits per case | Brier | Calibration slope |",
                  "|---|---|---|---|---|---|---|"]
        for r in rs:
            ci = cis[setting][r["model"]]
            lines.append(f"| {r['model']} | {r['auc']:.3f} ({ci['auc_ci'][0]:.3f}–{ci['auc_ci'][1]:.3f}) | "
                         f"{pct(r['cap_recall'])} ({pct(ci['recall_ci'][0])}–{pct(ci['recall_ci'][1])}) | {pct(r['cap_precision'])} | "
                         f"{r['cap_visits_per_case']:.1f} | {r.get('brier', float('nan')):.3f} | {r.get('cal_slope', float('nan')):.2f} |")
        lines += ["", f"![curves](curves-{setting.lower().replace(' ', '-').replace('/', '-')}.png)", ""]
    lines += ["## Read this before quoting the numbers", "",
              "- The thesis model was trained on the random split of IFLS5, so on these test sets it may have seen some of "
              "the children (the province hold-out especially). Its figures here are optimistic; the new models' are not.",
              "- Precision and visits per case depend on prevalence: they will differ in NTT, where stunting is more common.",
              "- The rule has only three score values, so its 'capacity' threshold flags whoever has at least that many risk factors; "
              "its flagged share may not equal the capacity exactly.",
              "- None of this is a clinical validation. For NTT, check the model on NTT children (recalibrate.py)."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("ml/out"))
    ap.add_argument("--label-col", default=None, help="thesis label column (low/medium/high); default: HAZ < -2")
    ap.add_argument("--weight-col", default=None, help="survey weight column, e.g. IFLS individual weight")
    ap.add_argument("--holdout-province", default=None, help="BPS province code to hold out, e.g. 52 (NTB)")
    ap.add_argument("--capacity", type=float, default=0.30, help="share of children Kader can visit (default 0.30)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, dtype={"district": str, "province": str, "hhid": str})
    df = df[df[args.label_col].notna()] if args.label_col else df[df["haz"].notna()]
    thesis = C.load_thesis_model()
    rows, cis, info = [], {}, {}

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(df, groups=df["hhid"]))
    name = "Random 80/20 split (by household)"
    r, ci, info[name] = run_setting(name, df.iloc[tr], df.iloc[te], args, thesis)
    rows += r
    cis[name] = ci
    if args.holdout_province:
        mask = df["province"] == str(args.holdout_province)
        name = f"Province {args.holdout_province} held out"
        r, ci, info[name] = run_setting(name, df[~mask], df[mask], args, thesis)
        rows += r
        cis[name] = ci

    # Final model on all the data, for use after the report is accepted.
    cols = sum(feature_lists(df), [])
    final = logistic(df, df["hhid"].to_numpy(), args.seed).fit(df[cols], C.make_target(df, args.label_col),
                                                                lr__sample_weight=C.weights(df, args.weight_col))
    p = final.predict_proba(df[cols])[:, 1]
    thr = C.threshold_for_capacity(p, C.weights(df, args.weight_col), args.capacity)
    joblib.dump({"pipeline": final, "inputs": cols, "threshold": thr, "capacity": args.capacity}, args.out / "model.joblib")
    (args.out / "metrics.json").write_text(json.dumps({"rows": rows, "ci": cis}, indent=1, default=float))
    (args.out / "report.md").write_text(report(rows, cis, info, args))
    print((args.out / "report.md").read_text())


if __name__ == "__main__":
    main()
