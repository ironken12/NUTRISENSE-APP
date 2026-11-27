// node parity_test.js : the app's scorer against scikit-learn on 300 synthetic children (ifls5-lr-2).
const fs = require("fs"); const { predictRisk } = require("./nutrisenseRisk.js");
const model = JSON.parse(fs.readFileSync(__dirname + "/nutrisense_risk_model.json"));
const { cases } = JSON.parse(fs.readFileSync(__dirname + "/parity_test_vectors.json"));
let maxDiff = 0, levelMismatch = 0;
for (const c of cases) {
  const r = predictRisk(model, c.child);
  maxDiff = Math.max(maxDiff, Math.abs(r.p_stunting - c.p_stunting));
  if (r.level !== c.level) levelMismatch++;
}
console.log(`cases ${cases.length}  max probability difference ${maxDiff.toExponential(2)}  level mismatches ${levelMismatch}`);
process.exit(maxDiff < 1e-9 && levelMismatch === 0 ? 0 : 1);
