const fs = require("fs"); const { predictRisk } = require("./nutrisenseRisk.js");
const model = JSON.parse(fs.readFileSync(__dirname + "/nutrisense_risk_model.json"));
const cases = JSON.parse(fs.readFileSync(__dirname + "/parity_test_vectors.json"));
let maxDiff = 0, levelMismatch = 0;
for (const c of cases) {
  const r = predictRisk(model, c.input); const p = [r.proba.low, r.proba.medium, r.proba.high];
  p.forEach((v, i) => { maxDiff = Math.max(maxDiff, Math.abs(v - c.proba[i])); });
  if (r.level !== c.level) levelMismatch++;
}
console.log(`cases ${cases.length}  max probability difference ${maxDiff.toExponential(2)}  level mismatches ${levelMismatch}`);
process.exit(maxDiff < 1e-9 && levelMismatch === 0 ? 0 : 1);
