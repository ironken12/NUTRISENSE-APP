"""The thesis risk model (app/ai/riskModel) as used by the app: parity with scikit-learn, inputs, guardrails, missing data."""
import json
import math
import shutil
import subprocess

import pytest

from app.ai import risk_model
from app.ai.nutrisense_risk import MODEL_DIR, load_model, predict_risk
from app.ai.risk_model import EXCLUDED_INPUTS, FEATURE_NAMES, RiskModel, clean_features

LEVELS = ["low", "medium", "high"]
# A child the model scores "low": tall, educated mother, normal birth weight, complete immunisation, urban household.
LOW = {"age_months": 20, "sex_male": 0, "birth_weight_g": 3600, "low_birth_weight": 0, "mother_height_cm": 160, "antenatal_contacts": 10,
       "mother_education": 4, "exclusive_bf_6m": 1, "diarrhoea_1m": 0, "ari_1m": 0, "immunisation_complete": 1, "mother_anaemia": 0,
       "improved_water": 1, "improved_sanitation": 1, "urban": 1, "district": "5371"}


@pytest.fixture(scope="module")
def model():
    return RiskModel()


def test_parity_with_scikit_learn():
    """The 300 parity vectors (scored by scikit-learn): max probability difference ~1e-16 and 0 level mismatches."""
    m = load_model()
    with open(MODEL_DIR / "parity_test_vectors.json", encoding="utf-8") as f:
        vectors = json.load(f)
    vectors = vectors["cases"] if isinstance(vectors, dict) else vectors
    max_diff, mismatches = 0.0, 0
    for case in vectors:
        out = predict_risk(m, case["input"])
        p = [out["proba"][k] for k in LEVELS]
        max_diff = max(max_diff, *(abs(a - b) for a, b in zip(p, case["proba"])))
        mismatches += out["level"] != case["level"]
    print(f"cases {len(vectors)}  max probability difference {max_diff:.3g}  level mismatches {mismatches}")
    assert len(vectors) == 300 and max_diff < 1e-12 and mismatches == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_parity_js_reference():
    """The unchanged JavaScript reference (node riskModel/parity_test.js) still passes."""
    r = subprocess.run(["node", "parity_test.js"], cwd=MODEL_DIR, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "level mismatches 0" in r.stdout


def test_model_file_is_the_thesis_model():
    m = load_model()
    assert m["version"] == risk_model.MODEL_VERSION == "ifls5-lr-1"
    assert math.isclose(m["threshold_high"], 0.2605571334742744)
    assert len(m["coef"]) == 3 and all(len(w) == len(m["feature_order"]) == 89 for w in m["coef"])


def test_child_with_no_data_returns_a_valid_level(model):
    p = model.predict({})
    assert p.risk_level in LEVELS and p.model_level in LEVELS
    assert math.isclose(sum(p.probabilities.values()), 1.0, abs_tol=1e-3)
    assert set(p.missing_inputs) == set(FEATURE_NAMES) and p.incomplete
    assert predict_risk(load_model(), {})["level"] in LEVELS  # the raw port as well


def test_height_and_size_fields_change_nothing(model):
    """The model never receives the child's height, weight, z-scores, MUAC, regional prevalence or 2T."""
    assert not EXCLUDED_INPUTS & set(FEATURE_NAMES)
    base = model.predict(LOW)
    for extra in ({"height_cm": 70.0}, {"height_cm": 95.0, "weight_kg": 9.0, "haz": -3.5, "whz": -3.2, "waz": -3.0, "muac_cm": 11.0},
                  {"regional_prevalence": 37.8}, {"weight_not_gaining": 2}):
        p = model.predict({**LOW, **extra})
        assert p.probabilities == base.probabilities and p.risk_level == base.risk_level
    raw = predict_risk(load_model(), LOW)
    assert predict_risk(load_model(), {**LOW, "height_cm": 70.0, "haz": -4}) == raw


def test_unknown_inputs_are_missing_never_zero():
    m = load_model()
    assert predict_risk(m, {"mother_height_cm": None}) == predict_risk(m, {})
    assert predict_risk(m, {"mother_height_cm": 0}) != predict_risk(m, {})
    clean, missing, rejected = clean_features({**LOW, "mother_height_cm": 30, "mother_education": 7, "district": "Kupang", "age_months": 75,
                                               "birth_weight_g": None, "urban": 0.5})
    assert set(rejected) == {"mother_height_cm", "mother_education", "district", "age_months", "urban"} and missing == ["birth_weight_g"]
    assert clean["mother_height_cm"] is None
    assert clean_features({**LOW, "age_months": 60.4})[0]["age_months"] == 60.0  # a few days past 5 years: clipped


def test_ntt_districts_are_unseen_by_the_model():
    m = load_model()
    cat = m["categorical"]
    assert not any(c.startswith("53") for c in cat["columns"] + cat["infrequent"])
    assert predict_risk(m, {**LOW, "district": "5371"}) == predict_risk(m, {**LOW, "district": None})


def test_model_level_then_guardrails_only_raise(model):
    assert model.predict(LOW).risk_level == "low"
    p = model.predict(LOW, danger_signs=["convulsions"])
    assert p.risk_level == "high" and p.guardrail == "danger_signs_present" and p.needs_review and p.model_level == "low"
    assert model.predict(LOW, who_status={"haz": -3.4}).risk_level == "high"
    assert model.predict(LOW, who_status={"oedema": True}).risk_level == "high"


@pytest.mark.parametrize("who,extra,two_t,level,guardrail", [
    # WHO 2006 / Permenkes 2/2020: WHZ < -2 wasted, < -3 severely wasted; HAZ < -3 severely stunted.
    ({"whz": -2.4}, {}, None, "medium", "wasted_minimum_medium"),
    ({"whz": -3.2}, {}, None, "high", "severe_stunting_or_wasting"),
    ({"haz": -3.1}, {}, None, "high", "severe_stunting_or_wasting"),
    ({"haz": -2.5}, {}, None, "medium", "stunted_minimum_medium"),
    # WHO 2013 / Kemenkes 2019: child MUAC (6-59 months) < 12.5 cm moderate, < 11.5 cm severe.
    ({"muac_cm": 12.0}, {}, None, "medium", "wasted_minimum_medium"),
    ({"muac_cm": 11.2}, {}, None, "high", "severe_stunting_or_wasting"),
    ({"muac_cm": 11.2}, {"age_months": 4}, None, "low", None),  # MUAC cut-offs apply from 6 months
    # Kemenkes 2020 (Buku KIA): 2T, no weight gain at two weighings in a row: a guardrail, not a model input.
    ({}, {}, 2, "medium", "two_t_minimum_medium"),
    ({}, {}, 1, "low", None),
])
def test_who_overrides(model, who, extra, two_t, level, guardrail):
    p = model.predict({**LOW, **extra}, who_status=who, weight_not_gaining=two_t)
    assert p.model_level == "low"
    assert p.risk_level == level and p.guardrail == guardrail
    if guardrail:
        assert p.overrides and p.needs_review


def test_guardrails_never_lower(model):
    sick = {**LOW, "birth_weight_g": 2200, "low_birth_weight": 1, "mother_height_cm": 143, "mother_education": 1, "immunisation_complete": 0,
            "improved_sanitation": 0, "urban": 0}
    assert model.predict(sick).model_level == "high"
    for who in ({"whz": -2.5, "haz": 0.5}, {"haz": -2.5}, {}, {"muac_cm": 12.0}):
        p = model.predict(sick, who_status=who, weight_not_gaining=2)
        assert p.risk_level == "high" and p.guardrail is None and not p.needs_review


def test_no_confidence_review_rule(model):
    """The old "confidence < 0.6" rule is gone: a complete record the model scores without a guardrail needs no review."""
    p = model.predict(LOW)
    assert p.confidence < 0.6 and not p.needs_review


def test_missing_key_inputs_trigger_review_only_when_they_could_change_the_level(model):
    unknown = {k: None for k in ("birth_weight_g", "low_birth_weight", "mother_height_cm", "mother_education")}
    p = model.predict({**LOW, **unknown})
    assert p.incomplete and p.missing_changes_level
    assert p.needs_review == (p.risk_level != "high")
    p = model.predict({**LOW, "mother_height_cm": 30})
    assert p.needs_review and p.rejected_inputs == ["mother_height_cm"]
    assert not model.predict(LOW).incomplete


def test_explanations_point_to_drivers(model):
    p = model.predict({**LOW, "birth_weight_g": 2200, "low_birth_weight": 1, "mother_height_cm": 143})
    assert p.risk_level == "high" and p.contributions[0]["contribution"] > 0
    assert {c["feature"] for c in p.contributions[:3]} & {"mother_height_cm", "birth_weight_g"}
    p = model.predict({**LOW, "mother_height_cm": None})
    assert all(c["feature"] != "mother_height_cm" for c in p.contributions)  # imputed inputs are not explained


def test_model_card_is_served_as_given():
    card = risk_model.model_card()
    assert set(card["evaluation"]) == {"Random 80/20 split", "NTB hold-out"}
    assert card["evaluation"]["NTB hold-out"]["recall_high"] == 0.821 and "Not a clinical validation" in card["caveats"]


def test_override_and_missing_data_reasons_are_plain(model):
    from app.services.assessment import reasons

    feats = {**LOW, "birth_weight_g": None, "low_birth_weight": None, "mother_height_cm": None, "mother_education": None}
    who = {"haz": -1.0, "whz": -2.4, "waz": -1.5, "muac_cm": 12.1, "oedema": False, "haz_velocity": 0.0}
    pred = model.predict(feats, who_status=who)
    facts = {"age_months": 20, "weight_not_gaining": 0, "diarrhea": 0, "fever": 0, "repeated_infection": 0, "poor_appetite": 0,
             "dietary_diversity": None, "animal_protein_days": None, "exclusive_breastfeeding": 1, "immunization_complete": 1,
             "clean_water": 1, "sanitation": 1, "low_birth_weight": None, "premature": None, "mother_kek": None, "mother_short": None}
    ctx = {"features": feats, "facts": facts, "who": who, "trend": {}, "danger_signs": []}
    codes = {r["code"]: r["text"] for r in reasons(ctx, "id", pred)}
    assert {"whz_wasted", "muac_low"} <= set(codes)
    key = "incomplete" if "incomplete" in codes else "incomplete_minor"
    assert codes[key].startswith("Data belum lengkap") and "tinggi ibu" in codes[key] and "berat lahir" in codes[key]
    if pred.risk_level != pred.model_level:
        assert "raised" in codes and "BB/TB di bawah -2 SD" in codes["raised"]


def test_app_records_map_to_model_inputs(client):
    """build_context maps the seeded records to the model's inputs: None when unknown, never 0 as a stand-in."""
    from datetime import timedelta

    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import AncVisit, Child, MaternalMeasurement, Pregnancy
    from app.services.assessment import build_context, district_code

    with SessionLocal() as db:
        adel = db.scalar(select(Child).where(Child.name == "Adel Fanggidae"))
        f = build_context(db, adel)["features"]
        assert set(f) == set(FEATURE_NAMES)
        assert f["district"] == "5303" and f["urban"] == 0.0  # Baumata, Kabupaten Kupang
        assert f["birth_weight_g"] == 3100 and f["low_birth_weight"] == 0.0 and f["sex_male"] == 0.0
        # Mother Maria's current pregnancy is not Adel's: height and education may come from it, ANC visits and Hb may not.
        assert f["mother_height_cm"] == 149.0 and f["mother_education"] == 3.0
        assert f["antenatal_contacts"] is None and f["mother_anaemia"] is None
        assert f["immunisation_complete"] == 1.0 and f["exclusive_bf_6m"] in (0.0, 1.0, None)
        assert abs(f["age_months"] - (adel.measurements[-1].measured_at - adel.birth_date).days / 30.4375) < 1e-9

        kristo = db.scalar(select(Child).where(Child.name == "Kristo Pello"))  # 3 months old
        f = build_context(db, kristo)["features"]
        assert f["exclusive_bf_6m"] is None and f["immunisation_complete"] is None  # too young to know
        assert f["district"] == "5314"

        # A pregnancy linked to the child: its ANC visits and Hb count.
        p = Pregnancy(mother_id=kristo.caregiver_id, hpht=kristo.birth_date - timedelta(days=280), mother_height_cm=147.0, education="smp",
                      status="delivered", child_id=kristo.id)
        db.add(p)
        db.flush()
        for n in range(1, 5):
            db.add(AncVisit(pregnancy_id=p.id, number=n, visit_date=p.hpht + timedelta(weeks=8 * n)))
        db.add(MaternalMeasurement(pregnancy_id=p.id, measured_at=p.hpht + timedelta(weeks=20), gestational_weeks=20, hb_g_dl=11.1))
        db.flush()
        f = build_context(db, kristo)["features"]
        assert f["antenatal_contacts"] == 4.0 and f["mother_anaemia"] == 1.0
        assert f["mother_height_cm"] == 147.0 and f["mother_education"] == 2.0
        db.rollback()

        class R:
            district = "Kota Kupang"
        assert district_code(R()) == "5371" and district_code(None) is None
