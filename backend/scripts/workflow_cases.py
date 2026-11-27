"""Run the four workflow cases of thesis Table 4.15 through the NutriSense API on a fresh demo database.

    python scripts/workflow_cases.py results.json      (from backend/; uses a temporary database, never nutrisense.db)

Each case: a new mother account (consent given), the synthetic profile entered through the normal API, then the final
measurement or check that triggers the assessment. Observed: risk level (model and final), guardrail, review flag,
triage, case, supply request / locker / pickup code, and the notifications each role received.
"""
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
tmp = Path(tempfile.mkdtemp(prefix="wf-"))
os.environ.update(NUTRISENSE_DATABASE_URL=f"sqlite:///{tmp / 'wf.db'}", NUTRISENSE_MODEL_DIR=str(tmp / "models"),
                  NUTRISENSE_ANTHROPIC_API_KEY="", NUTRISENSE_SEED_DEMO_DATA="true")
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app.ai import kia  # noqa: E402
from app.ai.growth import _lms, _value_at, compute_z_scores  # noqa: E402
from app.main import app  # noqa: E402

DPM = 30.4375
TODAY = date.today()
PW = "Demo1234!"
out = {"run_date": TODAY.isoformat(), "cases": []}


def login(c, email, pw=PW):
    r = c.post("/api/auth/login", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def register_mother(c, email, name, region_id):
    r = c.post("/api/auth/register", json={"email": email, "password": "Ibu12345", "full_name": name, "region_id": region_id,
                                           "consent_data_processing": True, "consent_ai_analysis": True})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def size_for(sex, age_m, haz, whz):
    """Height (cm) and weight (kg) that give exactly this HAZ and WHZ (WHO 2006 LMS tables, as the app uses)."""
    h = round(_value_at(*_lms("lfa" if age_m < 24 else "hfa", sex, age_m), haz), 1)
    w = round(_value_at(*_lms("wfl" if age_m < 24 else "wfh", sex, h), whz), 2)
    return h, w


def notes(c, h):
    return [n.get("title") or n.get("text") or n.get("body") for n in c.get("/api/notifications", headers=h).json()]


def child_case(c, *, label, name, sex, age_m, haz, whz, bw, gains, symptom=None, anc=None, region="Oesapa", family=None,
               mother=None):
    regions = {r["name"]: r["id"] for r in c.get("/api/regions").json()}
    email = f"wf.{name.split()[1].lower()}@test.id"
    mom = register_mother(c, email, f"Ibu {name.split()[1]}", regions[region])
    birth = TODAY - timedelta(days=round(age_m * DPM))
    child = c.post("/api/children", headers=mom, json={"name": name, "sex": sex, "birth_date": birth.isoformat(),
                                                       "birth_weight_kg": bw, "clean_water_access": True,
                                                       "sanitation_access": True}).json()
    cid = child["id"]
    setup = []
    if anc:  # A past pregnancy's antenatal visits cannot be entered through the app: inserted in the database (test setup).
        from app.database import SessionLocal
        from app.models import AncVisit, Child, Pregnancy
        with SessionLocal() as db:
            ch = db.get(Child, cid)
            p = Pregnancy(mother_id=ch.caregiver_id, region_id=ch.region_id, kader_id=ch.kader_id, hpht=birth - timedelta(days=280),
                          status="delivered", delivered_at=birth, child_id=cid, **(mother or {}))
            db.add(p)
            db.flush()
            for n in range(1, anc + 1):
                db.add(AncVisit(pregnancy_id=p.id, number=min(n, 6), visit_date=p.hpht + timedelta(weeks=6 * n)))
            db.commit()
        setup.append(f"{anc} antenatal visits on a delivered pregnancy linked to the child (inserted in the database)"
                     + (f", with {mother}" if mother else ""))
    if family:
        r = c.patch(f"/api/children/{cid}", headers=mom, json=family)
        assert r.status_code == 200, r.text
        setup.append(f"family details through the app: {family}")
    # Monthly weighings: the profile's HAZ/WHZ at the last one; earlier ones on the same z-scores; `gains` = weight change
    # (kg) at each step, None = the growth the z-scores give.
    # `gains` = weight gain (kg) between consecutive monthly weighings; the LAST weighing is exactly on the profile's
    # HAZ/WHZ and the earlier weights are built backwards from it, so the 2T pattern is exactly the one specified.
    k = len(gains) + 1
    ages = [age_m - (k - 1 - i) for i in range(k)]
    sizes = [size_for(sex, a, haz, whz) for a in ages]
    weights = [sizes[-1][1]]
    for g in reversed(gains):
        weights.insert(0, round(weights[0] - g, 2))
    if symptom:
        r = c.post(f"/api/children/{cid}/symptoms", headers=mom, json={"description": symptom})
        setup.append(f"symptom report: \"{symptom}\" (HTTP {r.status_code})")
    last = None
    for i, a in enumerate(ages):
        d = birth + timedelta(days=round(a * DPM))
        body = {"measured_at": d.isoformat(), "height_cm": sizes[i][0], "weight_kg": weights[i],
                "position": "lying" if a < 24 else "standing", "run_assessment": i == k - 1}
        r = c.post(f"/api/children/{cid}/measurements", headers=mom, json=body)
        assert r.status_code == 201, r.text
        last = r.json()
    a = last["assessment"]
    m = last["measurement"]
    pts = [(birth + timedelta(days=round(x * DPM)), x, w) for x, w in zip(ages, weights)]
    kader = login(c, "kader.oesapa@nutrisense.id")
    officer = login(c, "officer@nutrisense.id")
    cases = [x for x in c.get("/api/cases", headers=officer).json() if x["child_id"] == cid]
    reqs = [x for x in c.get("/api/supply-requests", headers=officer).json() if x.get("child_id") == cid]
    tri = a["triage"]
    out["cases"].append({
        "case": label, "child": name, "setup": setup,
        "measurement": {"age_months": round(m["age_months"], 1), "height_cm": m["height_cm"], "weight_kg": m["weight_kg"],
                        "haz": m["haz"], "whz": m["whz"], "waz": m["waz"]},
        "weighings_2t": [w["result"] for w in kia.weighings(pts)], "not_gaining": kia.not_gaining(pts),
        "model_level": a["model_risk_level"], "final_level": a["risk_level"], "p_stunting": a["probabilities"].get("stunting"),
        "guardrail": a["guardrail"], "needs_review": a["needs_review"],
        "missing_inputs": a["features"].get("imputed"),
        "triage": {"urgency": tri.get("urgency"), "actions": [x.get("text") for x in tri.get("actions", [])]},
        "reasons": [r["text"] for r in a["reasons"]],
        "cases": [{"id": x["id"], "status": x["status"], "priority": x.get("priority"), "urgency": x.get("urgency"),
                   "assigned_to_id": x.get("assigned_to_id")} for x in cases],
        "supply_requests": [{"id": x["id"], "status": x["status"], "items": [i.get("item_key") for i in x.get("items", [])],
                             "pickup_code": x.get("pickup_code"), "locker": (x.get("locker") or {}).get("name") if isinstance(x.get("locker"), dict) else x.get("locker_id"),
                             "route": x.get("route") or x.get("delivery_mode")} for x in reqs],
        "notifications": {"mother": notes(c, login(c, email, "Ibu12345")), "kader": notes(c, kader)[:6], "officer": notes(c, officer)[:6]},
    })


def pregnancy_case(c, *, label):
    regions = {r["name"]: r["id"] for r in c.get("/api/regions").json()}
    email = "wf.hamil@test.id"
    mom = register_mother(c, email, "Ibu Wf Hamil", regions["Oesapa"])
    p = c.post("/api/pregnancies", headers=mom, json={"gestational_weeks": 20, "mother_height_cm": 152, "education": "smp", "gravida": 2}).json()
    pid = p["id"]
    hpht = date.fromisoformat(p["hpht"])
    for n, w in ((1, 9), (2, 16)):
        r = c.post(f"/api/pregnancies/{pid}/anc", headers=mom, json={"number": n, "visit_date": (hpht + timedelta(weeks=w)).isoformat()})
        assert r.status_code == 201, r.text
    r = c.post(f"/api/pregnancies/{pid}/measurements", headers=mom, json={"muac_cm": 22.8, "hb_g_dl": 10.4})
    assert r.status_code == 201, r.text
    det = c.get(f"/api/pregnancies/{pid}", headers=mom).json()
    officer = login(c, "officer@nutrisense.id")
    kader = login(c, "kader.oesapa@nutrisense.id")
    reqs = [{"id": x["id"], "status": x["status"], "items": [i.get("item_key") for i in x.get("items", [])]}
            for x in c.get("/api/supply-requests", headers=officer).json() if x.get("pregnancy_id") == pid]
    out["cases"].append({
        "case": label, "setup": ["second trimester (20 weeks), 2 ANC visits, MUAC 22.8 cm and Hb 10.4 g/dL entered by the mother"],
        "gestational_weeks": det.get("gestational_weeks"), "risk": det.get("risk"), "flags": det.get("flags"),
        "next_visit": det.get("next_visit") or det.get("anc_next"), "measurement_result": r.json(),
        "supply_requests": reqs,
        "notifications": {"mother": notes(c, login(c, email, "Ibu12345")), "kader": notes(c, kader)[:6], "officer": notes(c, officer)[:6]},
        "facility_or_midwife": det.get("facility") or det.get("linked_facility"),
    })


with TestClient(app) as c:
    child_case(c, label="Low risk (4.9.1)", name="Wf Lowa", sex="female", age_m=18.0, haz=-0.5, whz=0.2, bw=3.1,
               gains=[0.25, 0.25, 0.25], anc=6, mother={"mother_height_cm": 157.0, "education": "sma"},
               family={"father_height_cm": 168, "mother_age_at_birth": 27, "birth_order": 2, "household_under5": 1,
                       "household_assets": ["own_house", "electricity", "fridge", "tv", "gas_cooking", "vehicle", "savings"]})
    child_case(c, label="Moderate risk (4.9.2)", name="Wf Moda", sex="male", age_m=10.0, haz=-1.6, whz=-0.5, bw=2.4,
               gains=[0.35, 0.35, 0.0], symptom="Anak diare sudah 3 hari minggu ini")
    child_case(c, label="High risk, child (4.9.3)", name="Wf Higha", sex="female", age_m=26.0, haz=-2.4, whz=-1.8, bw=None,
               gains=[0.3, 0.0, 0.0])
    pregnancy_case(c, label="High risk, pregnant woman (4.9.3)")

(Path(sys.argv[1]) if len(sys.argv) > 1 else tmp / "out.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
print(json.dumps(out, indent=1, ensure_ascii=False, default=str)[:20000])


# ---- Second step: the officer approves each case's supply request, as the workflow requires; then what the mother gets.
with TestClient(app) as c:
    officer = login(c, "officer@nutrisense.id")
    for case in out["cases"]:
        for req in case.get("supply_requests", []):
            if req.get("status") != "pending_approval":
                continue
            plan = c.get(f"/api/supply-requests/{req['id']}/plan", headers=officer).json()
            r = c.post(f"/api/supply-requests/{req['id']}/approve", headers=officer, json={})
            after = next(x for x in c.get("/api/supply-requests", headers=officer).json() if x["id"] == req["id"])
            email = {"Wf Lowa": "wf.lowa@test.id", "Wf Moda": "wf.moda@test.id", "Wf Higha": "wf.higha@test.id"}.get(case.get("child"), "wf.hamil@test.id")
            mom = login(c, email, "Ibu12345")
            mom = login(c, email, "Ibu12345")
            mine = next((x for x in c.get("/api/supply-requests", headers=login(c, email, "Ibu12345")).json() if x["id"] == req["id"]), {})
            case["after_approval_mother_view"] = {"pickup_code": mine.get("pickup_code"), "has_qr": bool(mine.get("qr_payload")),
                                                  "locker": (mine.get("locker") or {}).get("name"), "expires_at": mine.get("expires_at")}
            case["after_approval"] = {
                "approve_http": r.status_code, "plan_options": [o.get("mode") or o.get("kind") or o.get("label") for o in plan.get("options", [])][:4],
                "status": after["status"], "pickup_code": after.get("pickup_code"), "locker_id": after.get("locker_id"),
                "fields": sorted(after.keys()),
                "mother_notifications": notes(c, mom),
            }
    # Case 4: what the pregnancy page schedules next.
    mom = login(c, "wf.hamil@test.id", "Ibu12345")
    preg = c.get("/api/pregnancies", headers=mom).json()[0]
    det = c.get(f"/api/pregnancies/{preg['id']}", headers=mom).json()
    out["cases"][3]["pregnancy_detail_keys"] = sorted(det.keys())
    out["cases"][3]["pregnancy_next"] = {k: det.get(k) for k in det if any(s in k for s in ("next", "anc", "visit", "today", "plan", "risk", "due"))}

(Path(sys.argv[1]) if len(sys.argv) > 1 else tmp / "out.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
