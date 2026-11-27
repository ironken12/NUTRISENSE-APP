# NutriSense risk model for the app (same model as thesis Section 4.7): ARCHIVED

This is the thesis model **ifls5-lr-1**, kept unchanged for reproducibility. The app now runs **ifls5-lr-2** (see `../README.md`); `pytest tests/test_nutrisense_risk.py` and `node parity_test.js` in this folder still check this model.

- `nutrisense_risk_model.json`: the evaluated logistic regression (IFLS5, random split): imputation values, scaling, district columns, coefficients, threshold.
- `nutrisenseRisk.js`: `predictRisk(model, child)` returns `{proba: {low, medium, high}, level}`. No dependencies; works in Node and the browser.
- `model_card.json`: the figures for the app's model performance panel (same as thesis Table 4.10).
- `parity_test_vectors.json` + `parity_test.js`: 300 synthetic cases scored by scikit-learn. Run `node parity_test.js`: must print max difference ~1e-16 and 0 level mismatches.

Inputs exclude the child's height, weight and z-scores. Measured z-scores and the 2T rule still go through the triage rules (Table 4.9) after the model.
Wire-up in the app: load the JSON once, call predictRisk with the child's registration fields, show `level`, and replace the demo panel
with `model_card.json`. Then retake the Figure 4.15c screenshot.
