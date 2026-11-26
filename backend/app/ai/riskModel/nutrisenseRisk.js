// NutriSense risk model scorer: runs the exact logistic regression evaluated in thesis Section 4.7.
// Load nutrisense_risk_model.json and call predictRisk(model, child).
// child fields (null/undefined = missing): age_months, birth_weight_g, mother_height_cm, antenatal_contacts,
// mother_education (0-4), sex_male, low_birth_weight, exclusive_bf_6m, diarrhoea_1m, ari_1m, immunisation_complete,
// mother_anaemia, improved_water, improved_sanitation, urban (all 0/1), district (BPS code as text, e.g. "5371").
// The child's own height, weight and z-scores are deliberately NOT inputs.
function isMissing(v) { return v === null || v === undefined || (typeof v === "number" && Number.isNaN(v)); }

function featureVector(model, child) {
  const x = [];
  for (const f of model.numeric) {
    const v = isMissing(child[f.name]) ? f.impute : Number(child[f.name]);
    x.push((v - f.mean) / f.scale);
  }
  for (const f of model.binary) x.push(isMissing(child[f.name]) ? f.impute : Number(child[f.name]));
  const cat = model.categorical;
  let d = isMissing(child[cat.name]) ? cat.missing_value : String(child[cat.name]);
  if (cat.infrequent.includes(d)) d = "infrequent_sklearn";
  for (const c of cat.columns) x.push(c === d ? 1 : 0);   // unseen district -> all zeros
  return x;
}

function predictRisk(model, child) {
  const x = featureVector(model, child);
  const z = model.coef.map((w, k) => w.reduce((s, wi, i) => s + wi * x[i], model.intercept[k]));
  const m = Math.max(...z); const e = z.map(v => Math.exp(v - m)); const sum = e.reduce((a, b) => a + b, 0);
  const proba = e.map(v => v / sum);                        // [low, medium, high]
  const level = proba[2] >= model.threshold_high ? "high" : (proba[0] >= proba[1] ? "low" : "medium");
  return { proba: { low: proba[0], medium: proba[1], high: proba[2] }, level, modelVersion: model.version };
}

if (typeof module !== "undefined") module.exports = { predictRisk, featureVector };
