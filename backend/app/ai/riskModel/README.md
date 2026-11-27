# NutriSense risk model for the app: ifls5-lr-2

The model the app runs. It improves on the thesis model **ifls5-lr-1** (archived, unchanged, in [`ifls5-lr-1/`](ifls5-lr-1/)).
Trained and exported with `ml/export_app_model.py` from IFLS5; evaluation in [`ml/RESULTS.md`](../../../../ml/RESULTS.md).

| File | What it is |
|---|---|
| `nutrisense_risk_model.json` | Logistic regression for stunting (HAZ < -2): age spline knots, imputation values, scaling, coefficients, thresholds |
| `nutrisenseRisk.js` | `predictRisk(model, child)` returns `{p_stunting, level}`. No dependencies; Node and the browser |
| `model_card.json` | Figures for the app's model performance panel: both evaluation settings, comparisons, top inputs, caveats |
| `parity_test_vectors.json` + `parity_test.js` | 300 **synthetic** children scored by scikit-learn. `node parity_test.js`: max difference ~1e-16, 0 level mismatches |

**Inputs (20).** The thesis model's inputs without the district (every NTT district was unseen) plus five a Kader can ask:
age, sex, birth weight (and < 2,500 g), mother's height, education, antenatal visits and anaemia, exclusive
breastfeeding to 6 months, diarrhoea and cough with difficult breathing in the last 4 weeks, basic immunisation,
drinking water, toilet, urban/rural, **father's height, mother's age at the birth, birth order, children under 5 at home,
and a 10-item asset checklist**. Unknown values are `null` (imputed with the training median or most common value),
never 0. The child's height, weight and z-scores, the district, regional prevalence and 2T are **not** inputs.

**Levels.** high if P(stunting) ≥ 0.4218 (the top 30% of the training children: what Kader can visit), medium if
≥ 0.2837 (top 60%), otherwise low. The WHO / Kemenkes guardrails and 2T run after the model and only raise the level.

**Evaluation** (models trained without the test children; thresholds fixed on training data, as deployed):

| Setting | AUC (95% CI) | Recall at "high" (95% CI) | Precision | vs ifls5-lr-1 (AUC) | vs simple rule (AUC) |
|---|---|---|---|---|---|
| Random 80/20 split (n = 975) | 0.665 (0.624–0.707) | 43.9% (37.9%–49.9%) | 48.6% | +0.023 (-0.003 to +0.048) | +0.085 (+0.052 to +0.120) |
| NTB hold-out (n = 456) | 0.727 (0.679–0.769) | 54.4% (48.3%–60.8%) | 68.3% | +0.052 (+0.022 to +0.082) | +0.152 (+0.113 to +0.195) |

Not a clinical validation. Trained without NTT: recalibrate the thresholds on NTT children (`ml/recalibrate.py`).
A new version: run `ml/export_app_model.py`, then `node parity_test.js` and the backend tests.
