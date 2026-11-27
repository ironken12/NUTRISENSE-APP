import pytest

from app.ai import llm, nutrition, symptoms
from app.ai.triage import triage

# The risk model tests (parity with the thesis model, guardrails, missing data) are in test_nutrisense_risk.py.


@pytest.mark.parametrize("text,expected", [
    ("Anak saya mencret 3 hari dan tidak mau makan", {"diarrhea", "poor_appetite"}),
    ("demam tinggi dan kejang tadi malam", {"fever", "high_fever", "convulsions"}),
    ("batuk pilek, napas cepat", {"cough", "runny_nose", "fast_breathing"}),
    ("kaki bengkak dan makin kurus", {"oedema", "weight_loss"}),
    ("my child has diarrhea and won't drink", {"diarrhea", "unable_to_drink"}),
])
def test_symptom_lexicon(text, expected):
    assert expected <= set(symptoms.interpret_rules(text)["symptoms"])


# How mothers actually write: "-nya" suffixes, slang, word order, hyphens, English.
@pytest.mark.parametrize("text,expected", [
    ("badannya panas sejak kemarin dan napasnya cepat sekali", {"fever", "fast_breathing"}),
    ("napasnya cepet bgt", {"fast_breathing"}),
    ("cepat sekali napasnya dari tadi", {"fast_breathing"}),
    ("nafasnya berat dan ngos-ngosan", {"fast_breathing"}),
    ("anak saya susah bernafas", {"fast_breathing"}),
    ("tidak demam tapi sesak", {"fast_breathing"}),
    ("he is breathing very fast", {"fast_breathing"}),
    ("my daughter struggles to breathe", {"fast_breathing"}),
    ("dia rewel terus dan BAB-nya cair", {"diarrhea"}),
    ("pupnya encer 5x sehari", {"diarrhea"}),
    ("buang air besar terus dari pagi", {"diarrhea"}),
    ("pupnya encer dan ada darahnya", {"diarrhea", "bloody_stool"}),
    ("his poop is watery", {"diarrhea"}),
    ("kejangnya sudah berhenti tapi tadi lama", {"convulsions"}),
    ("tadi malam badannya kaku dan matanya mendelik", {"convulsions"}),
    ("anaknya lemas sekali dan susah dibangunkan", {"lethargy"}),
    ("dia gak mau makan, lemes bgt", {"poor_appetite", "lethargy"}),
    ("tidak mau makan dan minum sejak kemarin", {"poor_appetite", "unable_to_drink"}),
    ("sudah tidak mau menyusu", {"unable_to_drink"}),
    ("dia tidak mau ASI", {"unable_to_drink"}),
    ("muntah terus setiap habis minum", {"vomiting", "vomits_everything"}),
    ("kakinya bengkak dua-duanya", {"oedema"}),
    ("mukanya sembab", {"oedema"}),
    ("suhu 39,5 derajat", {"fever", "high_fever"}),
    ("demamnya tinggi sekali", {"fever", "high_fever"}),
    ("anak GTM seminggu ini", {"poor_appetite"}),
    ("panaaas badannya", {"fever"}),
])
def test_symptom_real_phrasings(text, expected):
    assert expected <= set(symptoms.interpret_rules(text)["symptoms"])


# Negated, resolved or unrelated statements must not raise a finding (least of all a danger sign).
@pytest.mark.parametrize("text,absent", [
    ("tidak demam, tidak sesak", {"fever", "fast_breathing"}),
    ("napasnya tidak cepat", {"fast_breathing"}),
    ("anak tidak napas cepat", {"fast_breathing"}),
    ("tidak pernah kejang", {"convulsions"}),
    ("no fever, eating well", {"fever", "poor_appetite"}),
    ("he is not breathing fast", {"fast_breathing"}),
    ("sudah tidak ASI lagi, sekarang minum susu sapi", {"unable_to_drink"}),
    ("sudah tidak menyusu, makannya banyak", {"unable_to_drink", "poor_appetite"}),
    ("susah minum obat", {"unable_to_drink"}),
    ("tidak rewel dan makan lahap", {"poor_appetite"}),
    ("demamnya sudah turun", {"fever"}),
    ("diarenya sudah sembuh", {"diarrhea"}),
    ("tidak ada batuk pilek", {"cough", "runny_nose"}),
    ("berat badannya naik, tidak kurus", {"weight_loss"}),
])
def test_symptom_negation_and_resolution(text, absent):
    assert not absent & set(symptoms.interpret_rules(text)["symptoms"])


@pytest.mark.parametrize("text,days", [
    ("mencret sejak kemarin", 1), ("demam dua hari", 2), ("batuk seminggu", 7), ("fever for 3 days", 3), ("diare dari kemarin lusa", 2),
])
def test_symptom_duration(text, days):
    assert symptoms.interpret_rules(text)["duration_days"] == days


def test_danger_sign_from_free_text_reaches_the_report():
    r = symptoms.interpret("badannya panas sejak kemarin dan napasnya cepat sekali")
    assert "fast_breathing" in r["danger_signs"] and r["interpreted_by"] == "rules"


def test_symptom_duration_and_danger():
    r = symptoms.interpret("BAB cair sudah dua hari, anak lemas sekali", lang="id")
    assert r["duration_days"] == 2
    assert "lethargy" in r["danger_signs"]
    assert r["interpreted_by"] == "rules"


def test_claude_output_is_merged_not_trusted_alone(monkeypatch):
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: {
        "symptoms": ["cough"], "duration_days": None, "appetite": "reduced", "summary": "Batuk.", "other_concerns": []})
    r = symptoms.interpret("anak kejang dan batuk")
    # Claude missed the convulsion; the rule layer still catches the danger sign.
    assert {"cough", "convulsions"} <= set(r["symptoms"])
    assert "convulsions" in r["danger_signs"]
    assert r["interpreted_by"] == "claude+rules"


def test_triage_tiers():
    z = {"haz": -2.4, "waz": -1.8, "whz": -1.0}
    assert triage("high", ["convulsions"], ["convulsions"], z, {}, 20)["urgency"] == "emergency"
    t = triage("high", [], ["diarrhea"], z, {"status": "declining"}, 20)
    assert t["urgency"] == "doctor_48h"
    assert {"ors_zinc", "pmt_biscuit"} <= {s["item_key"] for s in t["supplies"]}
    assert triage("low", [], [], {"haz": 0, "whz": 0}, {}, 20)["escalate"] is False
    assert triage("medium", [], [], {"haz": -1.5, "whz": -3.2}, {}, 30)["supplies"][0]["item_key"] == "rutf"


def test_meal_nutrients_and_diversity():
    items, totals, groups = nutrition.compute_meal([{"food_key": "telur"}, {"food_key": "nasi", "grams": 100}])
    assert totals["protein_g"] == pytest.approx(12.6 * 0.5 + 2.7, abs=0.1)
    assert set(groups) == {"eggs", "grains_roots"}


def test_recommendation_rule_based_offline():
    r = nutrition.recommend({"age_months": 14, "sex": "female", "haz": -2.2, "whz": -1.0, "symptoms": ["diarrhea"]},
                            {"gaps": ["iron_mg"], "dietary_diversity": 3, "mdd_met": False}, "high", "en")
    assert r["generated_by"] == "rules"
    assert r["recipes"] and all(x["min_age_months"] <= 14 for x in r["recipes"])
    assert any("ORS" in t for t in r["tips"])


def test_llm_disabled_without_key():
    assert llm.is_enabled() is False
    assert llm.complete_json("s", "u", {"type": "object"}) is None


def test_recipes_are_easy_and_use_everyday_foods():
    from app.ai.recipes import RECIPES

    assert len(RECIPES) >= 15
    for r in RECIPES:
        assert all(f in nutrition.FOODS for f in r["foods"]), r["key"]
        assert r["cost"] in (1, 2) and r["minutes"] <= 40
        assert len(r["steps"]["id"]) <= 3 and len(r["steps"]["en"]) == len(r["steps"]["id"])
        assert r["ingredients"]["id"] and len(r["ingredients"]["id"]) == len(r["ingredients"]["en"])
        # No expensive or instant foods in the menu suggester.
        assert not {"mie", "biskuit", "daging_sapi"} & set(r["foods"])


def test_menu_suggester_fills_missing_groups():
    r = nutrition.suggest_menus(14, ["grains_roots"], ["iron_mg"], "en", use_ai=False)
    assert len(r["suggestions"]) == 3
    added = {g for s in r["suggestions"] for g in s["adds_groups"]}
    assert len(added) >= 4  # suggestions are diverse, not three variants of the same dish
    assert all(s["min_age_months"] <= 14 and s["minutes"] <= 40 for s in r["suggestions"])
    assert nutrition.suggest_menus(4, [], [], "en", use_ai=False)["suggestions"] == []
    baby = nutrition.suggest_menus(7, [], [], "en", use_ai=False)
    assert all(s["min_age_months"] <= 7 for s in baby["suggestions"])


def test_eta_text_reads_as_hours():
    from app.services.logistics import eta_text
    assert eta_text(45.4) == "45 menit"
    assert eta_text(755) == "12 jam 35 menit"
    assert eta_text(120) == "2 jam"


def test_triage_uses_child_muac():
    t = triage("high", [], [], {"muac_cm": 11.0}, {}, 18)
    assert any(s["item_key"] == "rutf" for s in t["supplies"])
    t = triage("medium", [], [], {"muac_cm": 12.2}, {}, 18)
    assert any(s["item_key"] == "pmt_biscuit" and s["quantity"] == 2 for s in t["supplies"])
    assert not any(s["item_key"] == "rutf" for s in triage("low", [], [], {"muac_cm": 11.0}, {}, 4)["supplies"])


def test_cheap_food_list_is_cheap_and_local():
    """The "easy, cheap, at the market" list: good value for money, never beef or pork."""
    from app.ai.kitchen import PRICE_IDR

    for n in ("protein_g", "iron_mg", "vitamin_a_mcg", "zinc_mg", "calcium_mg", "energy_kcal"):
        foods = nutrition._foods_for(n)
        assert foods and not {"daging_sapi", "daging_babi", "mie", "biskuit"} & set(foods), (n, foods)
        assert all(PRICE_IDR.get(f, 1000) < nutrition.CHEAP_PORTION_IDR for f in foods)
    assert {"tempe", "ikan"} <= set(nutrition._foods_for("protein_g"))
    assert "daun_kelor" in nutrition._foods_for("iron_mg")


def test_ntt_recipes_are_quick_and_safe_for_babies():
    from app.ai.recipes import RECIPES

    by_key = {r["key"]: r for r in RECIPES}
    for key in ("catemak_jagung", "sup_ikan_kelor", "nasi_jagung_telur", "ubi_tumbuk_telur_kelor", "bubur_hati_labu"):
        assert key in by_key
    # Everything except the traditional family dish (jagung bose) is ready in 20 minutes.
    assert all(r["minutes"] <= 20 for r in RECIPES if r["key"] != "jagung_bose_ikan")
    # Recipes for babies under 1 year never add salt or sugar.
    for r in RECIPES:
        if r["min_age"] < 12:
            text = " ".join(r["ingredients"]["id"]).lower()
            assert "garam" not in text and "gula" not in text, r["key"]


def test_food_package_only_for_a_growth_problem():
    """A "medium" from the background risk model alone (normal growth) gets the Kader visit, not a food package."""
    normal = {"haz": -0.5, "waz": 0.0, "whz": 0.2}
    t = triage("medium", [], [], normal, {}, 18)
    assert t["urgency"] == "kader_7d" and t["supplies"] == []
    assert not any(a["code"] == "pickup_supplies" for a in t["actions"])
    # The same child with a weighing without enough gain (T), stunting or a falling growth line does get one.
    assert any(s["item_key"] == "pmt_biscuit" for s in triage("medium", [], [], normal, {}, 18, weight_not_gaining=1)["supplies"])
    assert any(s["item_key"] == "pmt_biscuit" for s in triage("medium", [], [], {**normal, "haz": -2.2}, {}, 18)["supplies"])
    assert any(s["item_key"] == "pmt_biscuit" for s in triage("medium", [], [], normal, {"status": "declining"}, 18)["supplies"])
    # Diarrhoea treatment is given whatever the growth.
    assert [s["item_key"] for s in triage("medium", [], ["diarrhea"], normal, {}, 18)["supplies"]] == ["ors_zinc"]
    # High risk always gets a package and a referral to the Puskesmas.
    t = triage("high", [], [], normal, {}, 18)
    assert t["supplies"] and t["urgency"] == "doctor_48h" and "Puskesmas" in t["actions"][0]["text"]
