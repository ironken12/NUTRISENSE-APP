"""Stunting risk classifier (classify() in the class diagram): the risk model, then the clinical guardrails.

1. The model: ifls5-lr-2, a logistic regression for stunting (HAZ < -2) with an age spline and missing-value
   indicators, trained on IFLS5 2014-15 (children aged 0-59 months, 13 provinces, no NTT) with ml/export_app_model.py.
   It improves on the thesis model ifls5-lr-1 (archived in riskModel/ifls5-lr-1/) with five inputs a Kader can collect:
   father's height, mother's age at the birth, birth order, children under 5 in the household and an asset checklist.
   Its coefficients, spline knots, imputation values, scaling and thresholds live in riskModel/nutrisense_risk_model.json
   and are scored by nutrisense_risk.predict_stunting, a port of riskModel/nutrisenseRisk.js (parity with
   scikit-learn about 1e-16, tests/test_nutrisense_risk.py). Nothing here changes that maths.
   Decision rule (from the JSON): high if P(stunting) >= threshold_high (top 30% of the training children, what Kader
   can visit), medium if >= threshold_medium (top 60%), else low.

   The child's height, weight, z-scores, MUAC, district, regional prevalence and 2T are deliberately NOT inputs:
   stunting is defined by height-for-age, so a model fed with size would only re-read the answer. Unknown inputs are
   passed as missing (never 0); the model imputes them and knows they were missing (missing-value indicators).

2. The guardrails. WHO / Kemenkes cut-offs (HAZ, WHZ, MUAC, oedema), 2T and danger signs are applied after the model
   and can only RAISE the level, never lower it.

The result is decision support for a health worker, not a diagnosis.

Every prediction ships with:
  * P(stunting) (stored as probabilities = {"stunting": p}; confidence = p),
  * per-input contributions (change in P(stunting) when that input is treated as unknown),
  * the guardrail that raised the level, if any,
  * a needs_review flag for the human-in-the-loop step,
  * the inputs that were missing or rejected.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .nutrisense_risk import load_model, load_model_card, predict_stunting

LABELS = ["low", "medium", "high"]
_MODEL = load_model()
MODEL_VERSION: str = _MODEL["version"]
ALGORITHM = "Logistic regression with age spline (ifls5-lr-2, IFLS5 2014-15); levels by P(stunting) thresholds"
THRESHOLD_HIGH: float = _MODEL["threshold_high"]
THRESHOLD_MEDIUM: float = _MODEL["threshold_medium"]

AGE = _MODEL["age"]["name"]
NUMERIC = [f["name"] for f in _MODEL["numeric"]]
BINARY = [f["name"] for f in _MODEL["binary"]]
DISTRICT = "district"  # not an input of ifls5-lr-2; kept for callers that pass it
FEATURE_NAMES: list[str] = [AGE] + NUMERIC + BINARY
# Never model inputs: the outcome itself, the child's size, area-level figures and the 2T weighing rule.
EXCLUDED_INPUTS = {"haz", "waz", "whz", "height_cm", "weight_kg", "haz_velocity", "waz_velocity", "regional_prevalence", "muac_cm",
                   "weight_not_gaining", "two_t", "district"}

# Possible range of each numeric input. A value outside it is a data error: it is rejected (treated as missing) and flagged.
RANGES: dict[str, tuple[float, float]] = {
    "age_months": (0.0, 60.0),
    "birth_weight_g": (500.0, 6000.0),
    "mother_height_cm": (120.0, 200.0),
    "father_height_cm": (130.0, 210.0),
    "antenatal_contacts": (0.0, 40.0),
    "mother_education": (0.0, 4.0),  # 0 none, 1 primary, 2 junior high, 3 senior high, 4 tertiary
    "maternal_age": (12.0, 60.0),  # mother's age at the child's birth, years
    "birth_order": (1.0, 20.0),  # order among the mother's pregnancies
    "n_under5": (1.0, 15.0),  # children under 5 living in the household, this child included
    "wealth_index": (0.0, 10.0),  # asset checklist, 0-10
}
RANGES.update({name: (0.0, 1.0) for name in BINARY})
AGE_CLIP_MONTHS = 61.0  # a check-up a few days after the 5th birthday is clipped to 60 months; beyond that it is an error

# The strongest inputs the app can be missing (model card top inputs, without age). If they are unknown and plausible
# values would give a different level, the result needs a human review.
KEY_INPUTS = ("father_height_cm", "mother_height_cm", "mother_education", "wealth_index")
_SCALE = dict(zip(_MODEL["scaling"]["columns"], zip(_MODEL["scaling"]["mean"], _MODEL["scaling"]["scale"])))


def _plausible(name: str) -> tuple[float, float]:
    """Low and high plausible value of a key input: about the 10th and 90th percentile (mean -/+ 1.28 sd) of the
    training data, within the input's range; the ends of the scale for education."""
    if name == "mother_education":
        return 1.0, 4.0
    mean, sd = _SCALE[name]
    lo, hi = RANGES[name]
    return max(lo, mean - 1.28 * sd), min(hi, mean + 1.28 * sd)


# WHO / Kemenkes cut-offs used by the guardrails (applied after the model, never as inputs).
#   WHZ < -2 wasted, < -3 severely wasted; HAZ < -2 stunted, < -3 severely stunted: WHO Child Growth Standards 2006;
#   Permenkes 2/2020 Standar Antropometri Anak (Kemenkes 2020).
#   Child MUAC (LiLA) 6-59 months: < 11.5 cm severe, < 12.5 cm moderate acute malnutrition: WHO 2013 guideline
#   "Updates on the management of severe acute malnutrition"; Kemenkes 2019 Pedoman Pencegahan dan Tata Laksana Gizi Buruk.
#   2T (no weight gain at two weighings in a row): Kemenkes 2020 (Buku KIA, KBM); a referral trigger for the Posyandu.
WHZ_WASTED, WHZ_SEVERE = -2.0, -3.0
HAZ_STUNTED, HAZ_SEVERE = -2.0, -3.0
MUAC_MODERATE_CM, MUAC_SEVERE_CM = 12.5, 11.5
MUAC_MIN_AGE_MONTHS = 6.0


def model_card() -> dict:
    return load_model_card()


def clean_features(features: dict) -> tuple[dict, list[str], list[str]]:
    """Range-check the model inputs. Returns (values with None for missing or rejected, missing names, rejected names).
    Anything that is not one of the model's inputs (height, weight, z-scores, 2T, ...) is ignored."""
    clean: dict = {}
    missing: list[str] = []
    rejected: list[str] = []
    for name in FEATURE_NAMES:
        v = features.get(name)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            clean[name] = None
            missing.append(name)
            continue
        if name == DISTRICT:
            s = str(v).strip()
            if s.isdigit() and len(s) == 4:  # a BPS kabupaten/kota code, e.g. "5371"
                clean[name] = s
            else:
                clean[name] = None
                rejected.append(name)
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = math.nan
        lo, hi = RANGES[name]
        if name == "age_months" and hi < v <= AGE_CLIP_MONTHS:
            v = hi
        if not math.isfinite(v) or v < lo or v > hi or (name in BINARY and v not in (0.0, 1.0)):
            clean[name] = None
            rejected.append(name)
            continue
        clean[name] = v
    return clean, missing, rejected


@dataclass
class Prediction:
    risk_level: str
    model_level: str
    probabilities: dict
    confidence: float
    contributions: list[dict] = field(default_factory=list)
    guardrail: str | None = None
    needs_review: bool = False
    overrides: list[str] = field(default_factory=list)  # every WHO/Kemenkes rule that applied
    missing_inputs: list[str] = field(default_factory=list)
    rejected_inputs: list[str] = field(default_factory=list)
    incomplete: bool = False  # a key input was missing or rejected
    missing_changes_level: bool = False  # plausible values for the missing key inputs give different levels


def who_overrides(who: dict, age_months: float | None, weight_not_gaining: float | None) -> tuple[list[str], list[str]]:
    """WHO / Kemenkes rules that set a minimum level: (rules for at least "high", rules for at least "medium")."""
    haz, whz, muac = who.get("haz"), who.get("whz"), who.get("muac_cm")
    muac_ok = muac is not None and (age_months is None or age_months >= MUAC_MIN_AGE_MONTHS)
    high, medium = [], []
    if who.get("oedema"):
        high.append("oedema")
    if whz is not None and whz < WHZ_SEVERE:
        high.append("whz_below_-3")
    if muac_ok and muac < MUAC_SEVERE_CM:
        high.append("muac_below_11.5")
    if haz is not None and haz < HAZ_SEVERE:
        high.append("haz_below_-3")
    if whz is not None and WHZ_SEVERE <= whz < WHZ_WASTED:
        medium.append("whz_below_-2")
    if muac_ok and MUAC_SEVERE_CM <= muac < MUAC_MODERATE_CM:
        medium.append("muac_below_12.5")
    if haz is not None and HAZ_SEVERE <= haz < HAZ_STUNTED:
        medium.append("haz_below_-2")  # to verify (clinical sign-off): HAZ -2..-3 keeps "at least medium"
    if weight_not_gaining is not None and weight_not_gaining >= 2:
        medium.append("2t")
    return high, medium


class RiskModel:
    """The risk model (loaded once from riskModel/nutrisense_risk_model.json) plus the guardrails."""

    def __init__(self, model_run_id: int | None = None):
        self.model = _MODEL
        self.model_run_id = model_run_id
        self.version = MODEL_VERSION

    def score(self, clean: dict) -> dict:
        """predictRisk() on the cleaned inputs only: {p_stunting, level, modelVersion}."""
        return predict_stunting(self.model, {k: clean.get(k) for k in FEATURE_NAMES})

    def predict(self, features: dict, danger_signs: list[str] | None = None, who_status: dict | None = None,
                weight_not_gaining: float | None = None) -> Prediction:
        """`features` are the model inputs (anything else is ignored). `who_status` (haz / whz / muac_cm / oedema from the
        latest measurement) and `weight_not_gaining` (2T) are used only by the guardrails after the model has spoken."""
        clean, missing, rejected = clean_features(features)
        out = self.score(clean)
        p = out["p_stunting"]
        model_idx = LABELS.index(out["level"])
        level_idx = model_idx

        # Explanation: how much each known input moves P(stunting), i.e. P with the child's value minus P with that
        # input unknown (the model's own imputation). Unknown inputs are left out: the model filled them in, not the child.
        deltas = []
        for name in FEATURE_NAMES:
            if clean[name] is None:
                continue
            d = p - self.score({**clean, name: None})["p_stunting"]
            deltas.append((name, clean[name], d))
        deltas.sort(key=lambda t: -abs(t[2]))
        contributions = [
            {"feature": n, "value": v if isinstance(v, str) else round(float(v), 3), "contribution": round(float(d), 4)}
            for n, v, d in deltas if abs(d) >= 0.01
        ][:8]

        # Clinical guardrails: danger signs and WHO/Kemenkes cut-offs (incl. 2T) can raise, never lower, the level.
        guardrail = None
        high_rules, medium_rules = who_overrides(who_status or {}, clean["age_months"], weight_not_gaining)
        if danger_signs:
            guardrail, level_idx = "danger_signs_present", 2
        elif high_rules:
            guardrail, level_idx = "severe_stunting_or_wasting", 2
        elif medium_rules and level_idx < 1:
            level_idx = 1
            if {"whz_below_-2", "muac_below_12.5"} & set(medium_rules):
                guardrail = "wasted_minimum_medium"
            elif "haz_below_-2" in medium_rules:
                guardrail = "stunted_minimum_medium"
            else:
                guardrail = "two_t_minimum_medium"

        confidence = float(p)
        # Missing data: a key input that is missing or was rejected makes the result "incomplete" (always shown as a plain
        # reason). It needs a human review when a value was rejected (a data error) or when plausible values for the
        # missing key inputs would give a different level, i.e. the level depends on data we do not have.
        unknown = set(missing) | set(rejected)
        unknown_key = [k for k in KEY_INPUTS if k in unknown]
        incomplete = bool(unknown_key) or bool(rejected)
        changes = False
        if unknown_key:
            levels = set()
            for combo in itertools.product(*(_plausible(k) for k in unknown_key)):
                trial = {**clean, **dict(zip(unknown_key, combo))}
                if "birth_weight_g" in unknown_key and clean.get("low_birth_weight") is None:
                    trial["low_birth_weight"] = float(trial["birth_weight_g"] < 2500)
                levels.add(self.score(trial)["level"])
            changes = len(levels) > 1
        needs_review = ((guardrail is not None and level_idx != model_idx) or bool(rejected) or (changes and level_idx < 2))
        return Prediction(
            risk_level=LABELS[level_idx],
            model_level=LABELS[model_idx],
            probabilities={"stunting": round(float(p), 4)},
            confidence=round(confidence, 4),
            contributions=contributions,
            guardrail=guardrail,
            needs_review=needs_review,
            overrides=(["danger_signs"] if danger_signs else []) + high_rules + medium_rules,
            missing_inputs=missing,
            rejected_inputs=rejected,
            incomplete=incomplete,
            missing_changes_level=changes,
        )
