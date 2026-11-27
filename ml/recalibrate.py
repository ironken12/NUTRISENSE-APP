"""Recalibrate the risk model's thresholds for NTT, without retraining.

    python ml/recalibrate.py --local ntt_children.csv --capacity 0.30 --out ml/out-ntt
    python ml/recalibrate.py --local ntt_children.csv --target-recall 0.80 --out ml/out-ntt
    python ml/recalibrate.py --model thesis --local ntt_children.csv ...      # the thesis model ifls5-lr-1 instead

The coefficients and scoring maths stay exactly as they are. Only the thresholds change, chosen on NTT children so that
either the share of children flagged high matches what Kader can visit (--capacity), or a target share of the truly
high-risk children is caught (--target-recall). For the active model (ifls5-lr-2) threshold_medium is also reset, to
flag the top --capacity-medium share as at least medium. Writes:
  * nutrisense_risk_model.json: the same model with the new thresholds and a new version name (e.g. ifls5-lr-2-ntt).
    It drops into backend/app/ai/riskModel/ with no code change (then update model_card.json and the parity vectors);
  * report.md: how the thesis model does on these NTT children, before and after, and its calibration there.

Input CSV: one row per NTT child with the thesis inputs (app names, see common.py), `hhid`, and `haz` or --label-col.
A few hundred children is the minimum; more is better. Optional --weight-col.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", required=True, type=Path, help="CSV of NTT children")
    ap.add_argument("--out", type=Path, default=Path("ml/out-ntt"))
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--weight-col", default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--capacity", type=float, default=None, help="share of children to flag high, e.g. 0.30")
    g.add_argument("--target-recall", type=float, default=None, help="share of high-risk children to catch, e.g. 0.80")
    ap.add_argument("--capacity-medium", type=float, default=0.60, help="active model: share flagged at least medium")
    ap.add_argument("--model", choices=["active", "thesis"], default="active")
    ap.add_argument("--version-suffix", default="ntt")
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    if args.capacity is None and args.target_recall is None:
        args.capacity = 0.30
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.local, dtype={"district": str, "province": str, "hhid": str})
    df = df[df[args.label_col].notna()] if args.label_col else df[df["haz"].notna()]
    y, w, groups = C.make_target(df, args.label_col), C.weights(df, args.weight_col), df["hhid"].to_numpy()
    if args.model == "thesis":
        model = C.load_thesis_model()
        p = C.thesis_p_high(df, model)
    else:
        model = C.load_thesis_model(C.ACTIVE_MODEL)
        p = C.active_p_stunting(df, model)

    old_thr = float(model["threshold_high"])
    if args.target_recall is not None:
        new_thr = C.threshold_for_recall(p, y, w, args.target_recall)
        rule = f"catch {100 * args.target_recall:.0f}% of high-risk children"
    else:
        new_thr = C.threshold_for_capacity(p, w, args.capacity)
        rule = f"flag the top {100 * args.capacity:.0f}% of children"
    before, after = C.at_threshold(p, y, w, old_thr), C.at_threshold(p, y, w, new_thr)
    auc = C.evaluate("thesis", p, y, w, 0.3)["auc"]
    ci = C.grouped_bootstrap(p, y, w, groups, 0.3, reps=args.bootstrap)
    cal_int, cal_slope = C.calibration(p, y)

    new = copy.deepcopy(model)
    new["threshold_high"] = new_thr
    if "threshold_medium" in model:
        new["threshold_medium"] = min(new_thr, C.threshold_for_capacity(p, w, args.capacity_medium))
    new["version"] = f"{model['version']}-{args.version_suffix}"
    new["threshold_note"] = (f"threshold_high recalibrated on {len(df)} local children to {rule}; "
                             f"coefficients unchanged from {model['version']}")
    (args.out / "nutrisense_risk_model.json").write_text(json.dumps(new, indent=1))

    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    row = lambda name, thr, r: (f"| {name} | {thr:.4f} | {pct(r['flagged'])} | {pct(r['recall'])} | {pct(r['precision'])} | "  # noqa: E731
                                f"{r['visits_per_case']:.1f} |")
    report = "\n".join([
        f"# Thesis model on local data: `{args.local.name}`", "",
        f"Children: {len(df)} · high risk ({'thesis label' if args.label_col else 'HAZ < -2'}): "
        f"{pct(np.average(y, weights=w))}", "",
        f"AUC of the model's score: **{auc:.3f}** (95% CI {ci['auc_ci'][0]:.3f}–{ci['auc_ci'][1]:.3f}). "
        f"Calibration intercept {cal_int:.2f}, slope {cal_slope:.2f} (ideal 0 and 1; a positive intercept means the model "
        "under-predicts risk here).", "",
        f"New threshold rule: {rule}.", "",
        "| Threshold | Score ≥ | Flagged high | Recall | Precision | Visits per case |", "|---|---|---|---|---|---|",
        row(f"Current ({model['version']})", old_thr, before), row(f"Recalibrated ({new['version']})", new_thr, after), "",
        "Only the threshold changed: the AUC (how well the model ranks children) is the same for both rows. If the AUC is "
        "low here, a new threshold cannot fix it; the model needs retraining with local data (train.py).", "",
        "To use it in the app: copy nutrisense_risk_model.json to backend/app/ai/riskModel/, update model_card.json "
        "(version, threshold, these figures), regenerate parity_test_vectors.json, and run the backend tests.",
    ]) + "\n"
    (args.out / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
