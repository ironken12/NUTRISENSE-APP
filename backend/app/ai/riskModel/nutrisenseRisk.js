// NutriSense risk model ifls5-lr-2: logistic regression with an age spline and missing-value indicators.
// Load nutrisense_risk_model.json and call predictRisk(model, child) -> { p_stunting, level, modelVersion }.
// child fields (null/undefined = unknown, never 0): age_months, birth_weight_g, mother_height_cm, antenatal_contacts,
// mother_education (0-4), maternal_age, birth_order, father_height_cm, n_under5, wealth_index (0-10), and 0/1:
// sex_male, low_birth_weight, exclusive_bf_6m, diarrhoea_1m, ari_1m, immunisation_complete, mother_anaemia,
// improved_water, improved_sanitation, urban. The child's own height, weight and z-scores are NOT inputs.
// Same arithmetic as backend/app/ai/nutrisense_risk.py (predict_stunting).
function isMissing(v) { return v === null || v === undefined || (typeof v === "number" && Number.isNaN(v)); }

// All B-spline basis values at x (Cox-de Boor, The NURBS Book A2.2), as scikit-learn's SplineTransformer.
function bsplineBasis(x, t, degree) {
  const n = t.length - degree - 1;
  let k = degree;
  while (k < n - 1 && x >= t[k + 1]) k++;
  const N = new Array(degree + 1).fill(0), left = new Array(degree + 1).fill(0), right = new Array(degree + 1).fill(0);
  N[0] = 1;
  for (let j = 1; j <= degree; j++) {
    left[j] = x - t[k + 1 - j]; right[j] = t[k + j] - x;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const temp = N[r] / (right[r + 1] + left[j - r]);
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    N[j] = saved;
  }
  const out = new Array(n).fill(0);
  for (let i = 0; i <= degree; i++) out[k - degree + i] = N[i];
  return out;
}

function featureVector(model, child) {
  const a0 = isMissing(child[model.age.name]) ? model.age.impute : Number(child[model.age.name]);
  const t = model.age.knots, d = model.age.degree;
  const a = Math.min(Math.max(a0, t[d]), t[t.length - d - 1]);   // constant extrapolation
  const x = bsplineBasis(a, t, d);
  const raw = model.numeric.map(f => isMissing(child[f.name]) ? f.impute : Number(child[f.name]));
  raw.forEach((v, i) => x.push((v - model.scaling.mean[i]) / model.scaling.scale[i]));
  for (const f of model.binary) x.push(isMissing(child[f.name]) ? f.impute : Number(child[f.name]));
  // Missing-value flags (0/1, not scaled); empty for the shipped model.
  for (const n of model.numeric_missing_indicators) x.push(isMissing(child[n]) ? 1 : 0);
  for (const n of model.binary_missing_indicators) x.push(isMissing(child[n]) ? 1 : 0);
  return x;
}

function predictRisk(model, child) {
  const x = featureVector(model, child);
  let z = model.intercept;
  model.coef.forEach((w, i) => { z = z + w * x[i]; });
  const p = z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  const level = p >= model.threshold_high ? "high" : (p >= model.threshold_medium ? "medium" : "low");
  return { p_stunting: p, level, modelVersion: model.version };
}

if (typeof module !== "undefined") module.exports = { predictRisk, featureVector, bsplineBasis };
