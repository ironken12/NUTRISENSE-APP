"""Build the child-level training table from the raw IFLS5 household files (RAND, hh14_all_dta.zip).

    python ml/build_ifls5.py --dta /path/to/unzipped/hh14_all_dta --out children.csv

The IFLS data is licensed by RAND for research use and must not be redistributed: keep the CSV out of git.

One row per child aged 0-59 months with a measured height, with the app's 16 model inputs (same definitions as
backend/app/services/assessment.py; null when unknown, never 0), the target (HAZ from the WHO standards, computed with
the app's own backend/app/ai/growth.py), the household id, province, and the IFLS5 cross-section health-measurement
weight. Child height and weight are only used for HAZ, never as inputs.

Sources (IFLS5 module: variables)
  roster BK_AR1: sex ar07, mother's line ar11, mother's education ar16 · location BK_SC1: province sc01, district sc02,
  urban sc05 · health measurement BUS_US + BUS_TIME: height us04, weight us06, position us05, Hb us13, date ·
  constructed birth date PTRACK: bth_day/month/year, weight pwt14usxa · pregnancy history B4_CH1 (mother's record for
  the child, ch27 = child's line): ANC visits ch16a-c, birth weight ch24, first water ch24c, first other food ch24d ·
  child morbidity B5_MAA2 (last 4 weeks): cough CA, difficulty breathing DA, diarrhoea IA · immunisation card B5_RJA3:
  dates of BCG, polio 1-3, DPT 1-3, measles · housing B2_KR: drinking water kr13, toilet kr20.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAYS_PER_MONTH = 30.4375


def _growth():
    spec = importlib.util.spec_from_file_location("growth", ROOT / "backend/app/ai/growth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rd(dta: Path, name: str, cols: list[str]) -> pd.DataFrame:
    return pd.read_stata(dta / name, columns=cols, convert_categoricals=False)


def edu_level(code) -> float:
    """IFLS ar16 -> model scale 0 none, 1 primary, 2 junior high, 3 senior high, 4 tertiary. Unknown -> NaN."""
    if pd.isna(code):
        return np.nan
    c = int(code)
    if c in (1, 90):
        return 0.0
    if c in (2, 11, 72):  # primary, Paket A, Madrasah Ibtidaiyah
        return 1.0
    if c in (3, 4, 12, 73):  # junior high, Paket B, Madrasah Tsanawiyah
        return 2.0
    if c in (5, 6, 15, 74):  # senior high, Paket C, Madrasah Aliyah
        return 3.0
    if c in (13, 60, 61, 62, 63):  # open university, diploma, S1-S3
        return 4.0
    return np.nan  # pesantren, special school, other, don't know


def months(value, unit) -> float:
    """Age at first water/food in months from IFLS value + unit (3 days, 4 weeks, 5 months)."""
    if pd.isna(value) or pd.isna(unit) or value >= 98:
        return np.nan
    return {3: value / DAYS_PER_MONTH, 4: value * 7 / DAYS_PER_MONTH, 5: float(value)}.get(int(unit), np.nan)


def build(dta: Path) -> pd.DataFrame:
    g = _growth()
    ar = rd(dta, "bk_ar1.dta", ["hhid14_9", "pid14", "pidlink", "ar07", "ar10", "ar11", "ar16", "ar01a", "ar09"])
    pt = rd(dta, "ptrack.dta", ["pidlink", "bth_day", "bth_month", "bth_year", "pwt14usxa"])
    us = rd(dta, "bus_us.dta", ["hhid14_9", "pid14", "us04", "us06", "us05", "us13"])
    ut = rd(dta, "bus_time.dta", ["hhid14_9", "ivwday", "ivwmth", "ivwyr"]).drop_duplicates("hhid14_9")
    sc = rd(dta, "bk_sc1.dta", ["hhid14_9", "sc01_14_14", "sc02_14_14", "sc05"])
    kr = rd(dta, "b2_kr.dta", ["hhid14_9", "kr13", "kr20", "kr03", "kr11", "kr23", "kr24", "kr24a"])
    hr = rd(dta, "b2_hr1.dta", ["hhid14_9", "hrtype", "hr01"])

    # Children with a height measurement and a birth date.
    kids = (us.merge(ar, on=["hhid14_9", "pid14"]).merge(pt, on="pidlink", how="left").merge(ut, on="hhid14_9", how="left"))
    kids = kids[kids["us04"].between(40, 130) & kids["ar07"].isin([1, 3])].copy()
    birth = pd.to_datetime(dict(year=kids.bth_year, month=kids.bth_month, day=kids.bth_day), errors="coerce")
    seen = pd.to_datetime(dict(year=kids.ivwyr, month=kids.ivwmth, day=kids.ivwday), errors="coerce")
    kids["age_months"] = (seen - birth).dt.days / DAYS_PER_MONTH
    kids = kids[kids["age_months"].between(0, 59.99)].copy()

    def z(r):
        try:
            zz = g.compute_z_scores("male" if r.ar07 == 1 else "female", r.age_months,
                                    r.us06 if r.us06 and r.us06 > 0 else 10.0, r.us04, "lying" if r.us05 == 3 else "standing")
            return zz.haz
        except ValueError:
            return np.nan
    kids["haz"] = kids.apply(z, axis=1)
    kids = kids[kids["haz"].between(-6, 6)].copy()  # WHO flags |HAZ| > 6 as implausible

    # Mother: line ar11 in the same household -> her education, height and Hb.
    mothers = ar[["hhid14_9", "pid14", "ar16"]].rename(columns={"pid14": "ar11", "ar16": "m_edu"})
    m_us = us[["hhid14_9", "pid14", "us04", "us13"]].rename(columns={"pid14": "ar11", "us04": "m_height", "us13": "m_hb"})
    kids = kids.merge(mothers, on=["hhid14_9", "ar11"], how="left").merge(m_us, on=["hhid14_9", "ar11"], how="left")

    # Extra inputs (optional in train.py): father's height (line ar10), mother's age now, children under 5 living here.
    f_us = us[["hhid14_9", "pid14", "us04"]].rename(columns={"pid14": "ar10", "us04": "f_height"})
    kids = kids.merge(f_us, on=["hhid14_9", "ar10"], how="left")
    m_age = ar[["hhid14_9", "pid14", "ar09"]].rename(columns={"pid14": "ar11", "ar09": "m_age_now"})
    kids = kids.merge(m_age, on=["hhid14_9", "ar11"], how="left")
    here = ar[ar["ar01a"].isin([1, 2, 5, 11])]
    u5 = here[here["ar09"] < 5].groupby("hhid14_9").size().rename("n_u5").reset_index()
    kids = kids.merge(u5, on="hhid14_9", how="left")

    # Pregnancy history: the mother's record for this child (ch27 = child's line).
    ch = rd(dta, "b4_ch1.dta", ["hhid14_9", "pid14", "ch27", "ch05", "ch16a", "ch16b", "ch16c", "ch14", "ch24", "ch24cx", "ch24c",
                                "ch24c1", "ch24dx", "ch24d", "ch24d1"])
    ch = ch[ch["ch27"].notna()].rename(columns={"pid14": "ar11", "ch27": "pid14"}).drop_duplicates(["hhid14_9", "ar11", "pid14"])
    kids = kids.merge(ch, on=["hhid14_9", "ar11", "pid14"], how="left")

    # Morbidity in the last 4 weeks (Book 5), one row per child x symptom.
    ma = rd(dta, "b5_maa2.dta", ["hhid14_9", "pid14", "maatype", "maa01"])
    ma["maatype"] = ma["maatype"].astype(str).str.strip().str.upper()
    sym = ma.pivot_table(index=["hhid14_9", "pid14"], columns="maatype", values="maa01", aggfunc="first").reset_index()
    kids = kids.merge(sym[["hhid14_9", "pid14"] + [c for c in ("CA", "DA", "IA") if c in sym.columns]],
                      on=["hhid14_9", "pid14"], how="left")

    # Immunisation card (Book 5): a year recorded for each basic vaccine.
    vacc = {"b": "BCG", "d": "Polio 1", "e": "Polio 2", "f": "Polio 3", "g": "DPT 1", "h": "DPT 2", "i": "DPT 3", "j": "Measles"}
    im = rd(dta, "b5_rja3.dta", ["hhid14_9", "pid14", "rja27x"] + [f"rja28ayr_{k}" for k in vacc])
    kids = kids.merge(im, on=["hhid14_9", "pid14"], how="left")

    # Asset score: a short checklist a Kader can ask (own house, electricity, fridge, TV, gas/electric cooking,
    # and owning a vehicle, savings, jewellery, land or another house). 0-10.
    own = hr[hr["hr01"] == 1].assign(hrtype=lambda d: d["hrtype"].astype(str).str.strip())
    own = own[own["hrtype"].isin(["B", "C", "E", "G", "J"])].groupby("hhid14_9").size().rename("n_assets").reset_index()
    kr = kr.merge(own, on="hhid14_9", how="left")
    kr["wealth_index"] = ((kr["kr03"] == 1).astype(int) + (kr["kr11"] == 1) + kr["kr23"].isin([1, 3]) + (kr["kr24a"] == 1)
                          + kr["kr24"].isin([1, 2]) + kr["n_assets"].fillna(0)).astype(float)
    kids = kids.merge(sc, on="hhid14_9", how="left").merge(kr, on="hhid14_9", how="left")
    return to_inputs(kids, vacc)


def to_inputs(k: pd.DataFrame, vacc: dict) -> pd.DataFrame:
    yes = lambda s: s.map({1: 1.0, 3: 0.0})  # noqa: E731  IFLS 1 yes / 3 no; anything else unknown
    out = pd.DataFrame({"hhid": k["hhid14_9"].astype(str), "province": k["sc01_14_14"].astype("Int64").astype(str)})
    out["district"] = (k["sc01_14_14"].astype("Int64").astype(str) + k["sc02_14_14"].astype("Int64").astype(str).str.zfill(2))
    out["age_months"] = k["age_months"]
    out["sex_male"] = (k["ar07"] == 1).astype(float)
    bw = k["ch24"].where((k["ch24"] >= 1) & (k["ch24"] < 9)) * 1000  # 9.x = missing codes
    out["birth_weight_g"] = bw
    out["low_birth_weight"] = (bw < 2500).astype(float).where(bw.notna())
    out["mother_height_cm"] = k["m_height"].where(k["m_height"].between(120, 190))
    anc = k[["ch16a", "ch16b", "ch16c"]].where(k[["ch16a", "ch16b", "ch16c"]] < 98)
    out["antenatal_contacts"] = anc.sum(axis=1, min_count=1).where(k["ch14"] != 3, 0.0)  # no check-up at all -> 0 visits
    out["mother_education"] = k["m_edu"].map(edu_level)

    # Exclusive breastfeeding to 6 months: nothing else before 6 months. Unknown under 6 months.
    water = [months(v, u) if x == 1 else (np.inf if x == 96 else np.nan) for x, v, u in zip(k.ch24cx, k.ch24c, k.ch24c1)]
    food = [months(v, u) if x == 1 else (np.inf if x == 96 else np.nan) for x, v, u in zip(k.ch24dx, k.ch24d, k.ch24d1)]
    first = np.fmin(np.asarray(water, float), np.asarray(food, float))
    known = ~(np.isnan(water) & np.isnan(food))
    ebf = np.where(np.asarray(first) < 6, 0.0, np.where(np.isinf(first) | (np.asarray(first) >= 6), 1.0, np.nan))
    out["exclusive_bf_6m"] = np.where((k["age_months"] >= 6).to_numpy() & known, ebf, np.nan)

    out["diarrhoea_1m"] = yes(k["IA"]) if "IA" in k else np.nan
    cough, breath = (yes(k["CA"]) if "CA" in k else np.nan), (yes(k["DA"]) if "DA" in k else np.nan)
    out["ari_1m"] = (cough * breath) if not np.isscalar(cough) else np.nan  # cough AND difficult breathing

    # Basic immunisation from the card: all eight dated. No card seen -> unknown. Under 12 months -> unknown.
    years = k[[f"rja28ayr_{c}" for c in vacc]]
    dated = years.notna() & (years >= 2000) & (years <= 2016)
    card_seen = years.notna().any(axis=1)
    out["immunisation_complete"] = dated.all(axis=1).astype(float).where(card_seen & (k["age_months"] >= 12))

    out["mother_anaemia"] = (k["m_hb"] < 12).astype(float).where(k["m_hb"].between(3, 20))
    out["improved_water"] = k["kr13"].isin([1, 2, 5, 10]).astype(float).where(k["kr13"].notna())
    out["improved_sanitation"] = (k["kr20"] == 1).astype(float).where(k["kr20"].notna())
    out["urban"] = (k["sc05"] == 1).astype(float).where(k["sc05"].notna())
    # Extra inputs (not in the thesis model).
    out["father_height_cm"] = k["f_height"].where(k["f_height"].between(130, 200))
    out["maternal_age"] = (k["m_age_now"] - k["age_months"] / 12).where(k["m_age_now"].between(12, 60))  # age at the birth
    out["birth_order"] = k["ch05"].where(k["ch05"].between(1, 20))  # order among the mother's pregnancies
    out["n_under5"] = k["n_u5"].astype(float)
    out["wealth_index"] = k["wealth_index"]
    out["haz"] = k["haz"].round(3)
    out["weight"] = k["pwt14usxa"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dta", required=True, type=Path, help="folder with the unzipped IFLS5 .dta files")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    df = build(a.dta)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} children, {df['hhid'].nunique()} households, stunted (HAZ < -2): {(df['haz'] < -2).mean():.1%}")
    print("missing share per input:")
    print(df.drop(columns=["hhid", "province", "haz", "weight"]).isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
