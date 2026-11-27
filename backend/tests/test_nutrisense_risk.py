"""The risk models (app/ai/riskModel) as used by the app: parity with scikit-learn, inputs, guardrails, missing data.

ifls5-lr-2 is the active model (riskModel/); ifls5-lr-1, the thesis model, is archived in riskModel/ifls5-lr-1/ and keeps
its own parity test.
"""
import json
import math
import shutil
import subprocess

import pytest

from app.ai import risk_model
from app.ai.nutrisense_risk import MODEL_DIR, THESIS_DIR, load_model, load_thesis_model, predict_risk, predict_stunting
from app.ai.risk_model import EXCLUDED_INPUTS, FEATURE_NAMES, KEY_INPUTS, RiskModel, clean_features

LEVELS = ["low", "medium", "high"]
# A child the model scores "low": tall parents, educated mother, good birth weight, assets, complete immunisation, town.
LOW = {"age_months": 20, "sex_male": 0, "birth_weight_g": 3600, "low_birth_weight": 0, "mother_height_cm": 160, "antenatal_contacts": 10,
       "mother_education": 4, "exclusive_bf_6m": 1, "diarrhoea_1m": 0, "ari_1m": 0, "immunisation_complete": 1, "mother_anaemia": 0,
       "improved_water": 1, "improved_sanitation": 1, "urban": 1, "district": "5371", "father_height_cm": 170, "maternal_age": 28,
       "birth_order": 2, "n_under5": 1, "wealth_index": 8}
SICK = {**LOW, "birth_weight_g": 2200, "low_birth_weight": 1, "mother_height_cm": 143, "father_height_cm": 152, "mother_education": 1,
        "immunisation_complete": 0, "improved_sanitation": 0, "urban": 0, "wealth_index": 2}


@pytest.fixture(scope="module")
def model():
    return RiskModel()


def test_parity_with_scikit_learn():
    """ifls5-lr-2: the 300 synthetic parity children scored by scikit-learn. Max difference ~1e-16, 0 level mismatches."""
    m = load_model()
    with open(MODEL_DIR / "parity_test_vectors.json", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    max_diff, mismatches = 0.0, 0
    for c in cases:
        out = predict_stunting(m, c["child"])
        max_diff = max(max_diff, abs(out["p_stunting"] - c["p_stunting"]))
        mismatches += out["level"] != c["level"]
    print(f"cases {len(cases)}  max probability difference {max_diff:.3g}  level mismatches {mismatches}")
    assert len(cases) == 300 and max_diff < 1e-12 and mismatches == 0
    assert {c["level"] for c in cases} == set(LEVELS)


def test_parity_thesis_model_archived():
    """ifls5-lr-1 (the thesis model) still scores exactly as scikit-learn did."""
    m = load_thesis_model()
    with open(THESIS_DIR / "parity_test_vectors.json", encoding="utf-8") as f:
        vectors = json.load(f)
    vectors = vectors["cases"] if isinstance(vectors, dict) else vectors
    max_diff, mismatches = 0.0, 0
    for case in vectors:
        out = predict_risk(m, case["input"])
        p = [out["proba"][k] for k in LEVELS]
        max_diff = max(max_diff, *(abs(a - b) for a, b in zip(p, case["proba"])))
        mismatches += out["level"] != case["level"]
    assert len(vectors) == 300 and max_diff < 1e-12 and mismatches == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("folder", [MODEL_DIR, THESIS_DIR])
def test_parity_js_reference(folder):
    """node parity_test.js passes for the active model and the archived thesis model."""
    r = subprocess.run(["node", "parity_test.js"], cwd=folder, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "level mismatches 0" in r.stdout


def test_model_file_is_ifls5_lr_2():
    m = load_model()
    assert m["version"] == risk_model.MODEL_VERSION == "ifls5-lr-2" and m["kind"] == "logistic_spline_v2"
    assert 0 < m["threshold_medium"] < m["threshold_high"] < 1
    assert len(m["coef"]) == len(m["columns"]) and "district" not in json.dumps(m["columns"])
    # No missing-value flags: an unknown answer counts as a typical value, never as "lower risk" (see ml/train.py).
    assert m["numeric_missing_indicators"] == [] and m["binary_missing_indicators"] == []
    assert {"father_height_cm", "maternal_age", "birth_order", "n_under5", "wealth_index"} <= set(FEATURE_NAMES)


def test_child_with_no_data_returns_a_valid_level(model):
    p = model.predict({})
    assert p.risk_level in LEVELS and p.model_level in LEVELS and 0 < p.probabilities["stunting"] < 1
    assert set(p.missing_inputs) == set(FEATURE_NAMES) and p.incomplete
    # No false reassurance: with nothing known the child is not "low", or a health worker checks it.
    assert p.risk_level != "low" or p.needs_review
    assert predict_stunting(load_model(), {})["level"] in LEVELS  # the raw port as well


def test_height_and_size_fields_change_nothing(model):
    """The model never receives the child's height, weight, z-scores, MUAC, district, regional prevalence or 2T."""
    assert not EXCLUDED_INPUTS & set(FEATURE_NAMES)
    base = model.predict(LOW)
    for extra in ({"height_cm": 70.0}, {"height_cm": 95.0, "weight_kg": 9.0, "haz": -3.5, "whz": -3.2, "waz": -3.0, "muac_cm": 11.0},
                  {"regional_prevalence": 37.8}, {"weight_not_gaining": 2}, {"district": "1201"}):
        p = model.predict({**LOW, **extra})
        assert p.probabilities == base.probabilities and p.risk_level == base.risk_level
    raw = predict_stunting(load_model(), LOW)
    assert predict_stunting(load_model(), {**LOW, "height_cm": 70.0, "haz": -4}) == raw


def test_unknown_inputs_are_missing_never_zero():
    m = load_model()
    assert predict_stunting(m, {"mother_height_cm": None}) == predict_stunting(m, {})
    assert predict_stunting(m, {"father_height_cm": 0}) != predict_stunting(m, {})
    clean, missing, rejected = clean_features({**LOW, "mother_height_cm": 30, "mother_education": 7, "age_months": 75,
                                               "birth_weight_g": None, "urban": 0.5, "wealth_index": 12, "birth_order": 0})
    assert set(rejected) == {"mother_height_cm", "mother_education", "age_months", "urban", "wealth_index", "birth_order"}
    assert missing == ["birth_weight_g"] and clean["mother_height_cm"] is None
    assert clean_features({**LOW, "age_months": 60.4})[0]["age_months"] == 60.0  # a few days past 5 years: clipped


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
    assert model.predict(SICK).model_level == "high"
    for who in ({"whz": -2.5, "haz": 0.5}, {"haz": -2.5}, {}, {"muac_cm": 12.0}):
        p = model.predict(SICK, who_status=who, weight_not_gaining=2)
        assert p.risk_level == "high" and p.guardrail is None and not p.needs_review


def test_no_confidence_review_rule(model):
    """A complete record the model scores without a guardrail needs no review, whatever the probability."""
    p = model.predict(LOW)
    assert not p.needs_review and not p.incomplete


def test_missing_key_inputs_trigger_review_only_when_they_could_change_the_level(model):
    unknown = {k: None for k in KEY_INPUTS}
    p = model.predict({**LOW, **unknown})
    assert p.incomplete and p.missing_changes_level
    assert p.needs_review == (p.risk_level != "high")
    p = model.predict({**LOW, "mother_height_cm": 30})
    assert p.needs_review and p.rejected_inputs == ["mother_height_cm"]
    assert not model.predict(LOW).incomplete


def test_explanations_point_to_drivers(model):
    p = model.predict(SICK)
    assert p.risk_level == "high" and p.contributions[0]["contribution"] > 0
    assert {c["feature"] for c in p.contributions[:3]} & {"mother_height_cm", "father_height_cm", "birth_weight_g"}
    p = model.predict({**LOW, "mother_height_cm": None})
    assert all(c["feature"] != "mother_height_cm" for c in p.contributions)  # imputed inputs are not explained


def test_model_card_is_served_as_given():
    card = risk_model.model_card()
    assert set(card["evaluation"]) == {"Random 80/20 split", "NTB hold-out"}
    for ev in card["evaluation"].values():
        assert 0.5 < ev["auc"] < 1 and ev["auc_ci_low"] < ev["auc"] < ev["auc_ci_high"]
        assert {"ifls5-lr-1", "rule"} == set(ev["compared_with"]) and ev["compared_with"]["rule"]["ci_low"] > 0
    assert "Not a clinical validation" in card["caveats"] and len(card["top_inputs"]) == 5


def test_override_and_missing_data_reasons_are_plain(model):
    from app.services.assessment import reasons

    feats = {**LOW, "mother_height_cm": None, "father_height_cm": None, "mother_education": None, "wealth_index": None}
    who = {"haz": -1.0, "whz": -2.4, "waz": -1.5, "muac_cm": 12.1, "oedema": False, "haz_velocity": 0.0}
    pred = model.predict(feats, who_status=who)
    facts = {"age_months": 20, "weight_not_gaining": 0, "diarrhea": 0, "fever": 0, "repeated_infection": 0, "poor_appetite": 0,
             "dietary_diversity": None, "animal_protein_days": None, "exclusive_breastfeeding": 1, "immunization_complete": 1,
             "clean_water": 1, "sanitation": 1, "low_birth_weight": None, "premature": None, "mother_kek": None, "mother_short": None}
    ctx = {"features": feats, "facts": facts, "who": who, "trend": {}, "danger_signs": []}
    codes = {r["code"]: r["text"] for r in reasons(ctx, "id", pred)}
    assert {"whz_wasted", "muac_low"} <= set(codes)
    key = "incomplete" if "incomplete" in codes else "incomplete_minor"
    assert codes[key].startswith("Data belum lengkap") and "tinggi ibu" in codes[key] and "tinggi ayah" in codes[key]
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
        assert set(f) == set(FEATURE_NAMES) | {"district"}  # district is recorded but is not an input of ifls5-lr-2
        assert f["district"] == "5303" and f["urban"] == 0.0  # Baumata, Kabupaten Kupang
        assert f["birth_weight_g"] == 3100 and f["low_birth_weight"] == 0.0 and f["sex_male"] == 0.0
        # Mother Maria's current pregnancy is not Adel's: height and education may come from it, ANC visits and Hb may not.
        assert f["mother_height_cm"] == 157.0 and f["mother_education"] == 3.0
        assert f["antenatal_contacts"] is None and f["mother_anaemia"] is None
        assert f["immunisation_complete"] == 1.0 and f["exclusive_bf_6m"] in (0.0, 1.0, None)
        assert abs(f["age_months"] - (adel.measurements[-1].measured_at - adel.birth_date).days / 30.4375) < 1e-9
        # Family questions: from the child record; Maria's current pregnancy is not Adel's, so its gravida does not count.
        assert f["father_height_cm"] == 168.0 and f["maternal_age"] == 27.0 and f["birth_order"] == 2.0
        assert f["n_under5"] == 1.0 and f["wealth_index"] == 7.0

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
        assert f["birth_order"] == 4.0  # no gravida on that pregnancy: the child record's answer
        p.gravida = 3
        db.flush()
        assert build_context(db, kristo)["features"]["birth_order"] == 3.0  # the linked pregnancy's gravida wins
        kristo.household_assets = []
        kristo.household_under5 = None
        db.flush()
        f = build_context(db, kristo)["features"]
        assert f["wealth_index"] == 0.0 and f["n_under5"] is None  # asked and owns none -> 0; not asked -> unknown
        db.rollback()

        class R:
            district = "Kota Kupang"
        assert district_code(R()) == "5371" and district_code(None) is None


def test_demo_data_shows_every_level_and_a_review(client):
    """The demo data must not all score high: storyline children with recorded mothers score low, and Selfi (model low,
    weight-for-height below -2) is raised by the WHO guardrail and waits for a human review."""
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import Child, RiskAssessment

    with SessionLocal() as db:
        first = {}
        for a in db.scalars(select(RiskAssessment).order_by(RiskAssessment.id)).all():
            first.setdefault(a.child_id, a)  # the seeded assessment, before other tests add or review any
        by_name = {db.get(Child, cid).name: a for cid, a in first.items()}
        for name in ("Adel Fanggidae", "Grace Pello", "Kristo Pello"):
            assert by_name[name].features["model_level"] == "low", name
        selfi = by_name["Selfi Manafe"]
        assert selfi.features["model_level"] == "low" and selfi.risk_level == "medium"
        assert selfi.guardrail == "wasted_minimum_medium" and selfi.needs_review
        assert {a.risk_level for a in first.values()} == {"low", "medium", "high"}


def test_ari_is_cough_with_difficult_breathing(client):
    """ari_1m as in IFLS5 / the model: cough AND difficult or fast breathing in the last 4 weeks; unknown without reports."""
    from types import SimpleNamespace

    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import Child
    from app.services.assessment import model_inputs

    with SessionLocal() as db:
        adel = db.scalar(select(Child).where(Child.name == "Adel Fanggidae"))
        latest = adel.measurements[-1]
        ari = lambda *syms: model_inputs(db, adel, latest, [SimpleNamespace(symptoms=list(syms))], {}, [])["ari_1m"]  # noqa: E731
        assert ari("cough", "fast_breathing") == 1.0
        assert ari("fast_breathing") == 0.0 and ari("cough") == 0.0
        assert model_inputs(db, adel, latest, [], {}, [])["ari_1m"] is None


def test_family_questions_api(client, auth):
    """Family questions: saved at registration and by PATCH, validated, and unknown stays null."""
    h = auth("kader.oesapa@nutrisense.id")
    from datetime import date, timedelta

    birth = (date.today() - timedelta(days=400)).isoformat()
    body = {"name": "Family Child", "sex": "female", "birth_date": birth, "caregiver_phone": "081300000005",
            "father_height_cm": 165, "mother_age_at_birth": 25, "birth_order": 2, "household_under5": 2,
            "household_assets": ["tv", "electricity", "tv"]}
    r = client.post("/api/children", headers=h, json=body)
    assert r.status_code == 201, r.text
    c = r.json()
    assert c["father_height_cm"] == 165 and c["household_assets"] == ["electricity", "tv"] and c["birth_order"] == 2
    r = client.patch(f"/api/children/{c['id']}", headers=h, json={"father_height_cm": None, "household_assets": []})
    assert r.status_code == 200 and r.json()["father_height_cm"] is None and r.json()["household_assets"] == []
    assert client.patch(f"/api/children/{c['id']}", headers=h, json={"household_assets": ["boat"]}).status_code == 422
    assert client.patch(f"/api/children/{c['id']}", headers=h, json={"father_height_cm": 80}).status_code == 422


def test_indonesian_texts_use_a_decimal_comma():
    """Mothers read Indonesian numbers: "TB/U -1,9" and "1,5 dari 8", while English keeps "-1.9"."""
    from app.services.assessment import _reason

    assert "TB/U -1,9" in _reason("haz_risk", "id", -1.9)["text"] and "HAZ -1.9" in _reason("haz_risk", "en", -1.9)["text"]
    assert "1,5 dari 8" in _reason("diversity", "id", 1.5)["text"]
    assert "12,1 cm (di bawah 12,5 cm" in _reason("muac_low", "id", 12.1)["text"]
