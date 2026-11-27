# Risk model training and evaluation

Separate from the app: nothing here runs in the backend. Results: [RESULTS.md](RESULTS.md).

| Script | What it does |
|---|---|
| `build_ifls5.py` | Builds the child table (app input names) from the raw IFLS5 `.dta` files |
| `train.py` | Retrains (improved logistic regression, gradient boosting) and compares with the thesis model and a simple rule; writes a report, figures and `model.joblib` |
| `compare.py` | Paired bootstrap: is one model's AUC really higher than another's on the same children? |
| `recalibrate.py` | Resets the thesis model's threshold on NTT children; writes a model JSON the app can load unchanged |
| `make_synthetic.py` | Synthetic data for testing the scripts. Its numbers mean nothing |

```bash
pip install -r ml/requirements.txt
python ml/build_ifls5.py --dta /path/to/hh14_all_dta --out children.csv
python ml/train.py --data children.csv --holdout-province 52 --weight-col weight --out ml/out
python ml/compare.py --data children.csv --base-drop-extras --holdout-province 52 --weight-col weight
python ml/recalibrate.py --local ntt_children.csv --capacity 0.30 --out ml/out-ntt   # when NTT data is available
pytest ml   # smoke test on synthetic data
```

**IFLS data licence:** RAND licenses IFLS for research and does not allow redistribution. Keep the `.dta` files and
any CSV built from them out of git (`.gitignore` blocks `*.dta`, `ml/out*/` and `ml/data/`).

The app's model (`backend/app/ai/riskModel/`) is unchanged. A new logistic regression with the extra inputs would need
the app's input mapping and port extended (splines, missing indicators, new fields); `recalibrate.py` output needs no
code change.
