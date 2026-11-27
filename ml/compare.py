"""Paired comparison of models on the same test children: is model A's AUC really higher than model B's?

    python ml/compare.py --data children_extra.csv --data-base children.csv --holdout-province 52 --weight-col weight

Resamples households (1000x by default) and recomputes the AUC difference on each resample, so both models are
always judged on the same children. Reports the mean difference, its 95% CI and the share of resamples where A is not
better (a one-sided bootstrap p-value). Uses the same splits and seed as train.py.

--data has the extra inputs; --data-base is the same children with the thesis inputs only (build_ifls5.py output
before the extras were added, or the same file with the extra columns dropped: see --base-drop-extras).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import common as C
import train as T


def paired(a, b, y, w, g, reps, seed=7):
    rng = np.random.default_rng(seed)
    u = np.unique(g)
    ix = {k: np.flatnonzero(g == k) for k in u}
    d = []
    for _ in range(reps):
        i = np.concatenate([ix[k] for k in rng.choice(u, len(u))])
        d.append(roc_auc_score(y[i], a[i], sample_weight=w[i]) - roc_auc_score(y[i], b[i], sample_weight=w[i]))
    d = np.asarray(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()


def fit_score(train, test, args):
    cols = sum(T.feature_lists(train), [])
    lr = T.logistic(train, train["hhid"].to_numpy(), args.seed).fit(
        train[cols], C.make_target(train, args.label_col), lr__sample_weight=C.weights(train, args.weight_col))
    return lr.predict_proba(test[cols])[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--data-base", type=Path, default=None)
    ap.add_argument("--base-drop-extras", action="store_true", help="build the base inputs by dropping the extra columns")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--weight-col", default=None)
    ap.add_argument("--holdout-province", default=None)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dt = {"district": str, "province": str, "hhid": str}
    df = pd.read_csv(args.data, dtype=dt)
    if args.data_base:
        base = pd.read_csv(args.data_base, dtype=dt)
    else:
        base = df.drop(columns=[c for c in C.EXTRA_NUMERIC + C.EXTRA_BINARY if c in df.columns])
    keep = df[args.label_col].notna() if args.label_col else df["haz"].notna()
    df, base = df[keep], base[keep]
    thesis = C.load_thesis_model()

    settings = []
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(df, groups=df["hhid"]))
    settings.append(("Random 80/20 split (by household)", df.index[tr], df.index[te]))
    if args.holdout_province:
        m = df["province"] == str(args.holdout_province)
        settings.append((f"Province {args.holdout_province} held out", df.index[~m], df.index[m]))

    for name, itr, ite in settings:
        test = df.loc[ite]
        y, w, g = C.make_target(test, args.label_col), C.weights(test, args.weight_col), test["hhid"].to_numpy()
        s = {"new LR + extra inputs": fit_score(df.loc[itr], test, args),
             "new LR, thesis inputs": fit_score(base.loc[itr], base.loc[ite], args),
             "thesis model": C.thesis_p_high(test, thesis), "rule": T.rule_score(test)}
        print(f"## {name} (n = {len(test)})\n")
        print("| Comparison | AUC difference | 95% CI | Share of resamples ≤ 0 |\n|---|---|---|---|")
        for a, b in (("new LR + extra inputs", "thesis model"), ("new LR, thesis inputs", "thesis model"),
                     ("new LR + extra inputs", "new LR, thesis inputs"), ("thesis model", "rule"), ("new LR + extra inputs", "rule")):
            mu, lo, hi, p = paired(s[a], s[b], y, w, g, args.reps)
            print(f"| {a} − {b} | {mu:+.3f} | {lo:+.3f} to {hi:+.3f} | {p:.3f} |")
        print()


if __name__ == "__main__":
    main()
