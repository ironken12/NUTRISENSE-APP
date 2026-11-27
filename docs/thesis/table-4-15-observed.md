---
title: "Table 4.15 Workflow Test Cases: observed results"
---

# How the tests were run

- **Date:** 27 November 2026.
- **App version:** risk model ifls5-lr-2, branch `claude/sharp-bohr-t1b7pz`.
- **Script:** `backend/scripts/workflow_cases.py`, which reruns everything below.
- **Database:** a fresh demo database with the demo villages, lockers, Kader, officer and doctor accounts.
- **Accounts and data entry:** each case used a new mother account (consent given). The synthetic profile went in
  through the app's normal API, then the last measurement (or the mother's check-up values) triggered the system.
- **Measurements:** heights and weights were computed from the WHO 2006 tables so that the last measurement gives the
  profile's HAZ and WHZ. Earlier monthly weighings were set to give the profile's weight-gain pattern (N = gained
  enough, T = did not gain, by the Buku KIA minimum).
- **What was not specified:** the profiles do not give the family details (father's height, mother's height and
  education, mother's age, birth order, children under 5, assets), so they were left unknown, as for a newly
  registered family. For the moderate-risk boy and the pregnant woman, values the table does not give (WHZ −0.5;
  20 weeks, mother's height 152 cm) were chosen as neutral.
- **Database shortcut:** the app has no screen for entering a past pregnancy's antenatal visits. For the low-risk case
  ("mother had six antenatal contacts"), a delivered pregnancy with 6 visits was therefore linked to the child
  directly in the database.
- **Officer approval:** supply requests in the app wait for an officer's approval. The officer approval was carried
  out in the test, as it would be in practice.

# Table 4.15 with the Observed column

| Case | Synthetic profile | Expected level | Expected system path | Observed |
|---|---|---|---|---|
| Low risk (4.9.1) | Girl, 18 months; HAZ −0.5, WHZ +0.2; steady weight gain; birth weight 3,100 g; mother had six antenatal contacts | Low | Growth chart updated; age-appropriate feeding education; next routine visit scheduled; no supply request | **Medium** (not as expected). HAZ −0.48, WHZ +0.20, weighings N-N-N. Model P(stunting) 0.31, just above the medium threshold of 0.28, because the mother's height and education and the family details were unknown. Flagged for health-worker review ("data belum lengkap", data incomplete). A Kader visit was scheduled within 7 days, a case was opened, and a supply request (PMT biscuits, Taburia, moringa powder) was created. After officer approval it was reserved at Loker Posyandu Oesapa, and the mother was sent a pickup code and QR |
| Moderate risk (4.9.2) | Boy, 10 months; HAZ −1.6; no weight gain at the last weighing; birth weight 2,400 g; diarrhoea in the past month | Medium | Case opened for Kader follow-up at the next session; nutrition guidance; supplementary food reserved in the locker; QR pickup code sent | **Medium** (as expected). HAZ −1.61, weighings N-N-T. Reasons shown: height approaching the stunting line, no weight gain at the last weighing, diarrhoea, low birth weight, incomplete data (flagged for review). Case opened for a Kader visit within 7 days. Guidance: ORS and zinc for 10 days, and the weekly nutrition plan. Supply request (ORS + zinc, PMT, Taburia, moringa powder): after officer approval, reserved at Loker Posyandu Oesapa; the mother received "Paket gizi siap diambil" with a 6-digit pickup code and QR, valid 3 days |
| High risk, child (4.9.3) | Girl, 26 months; HAZ −2.4, WHZ −1.8; no weight gain at two consecutive weighings | High | Kader home visit within 7 days; referral to puskesmas; stock check; simulated drone dispatch if locker stock is below threshold; officer alerted | **Medium** (not as expected). HAZ −2.38, WHZ −1.79, WAZ −2.66, weighings N-T-T (2T). Model P(stunting) 0.32. The WHO rules (stunted, 2T) set a minimum of medium, and no rule raises this profile to high: high requires HAZ or WHZ < −3, MUAC < 11.5 cm, oedema or a danger sign. Kader home visit within 7 days: yes. Officer alerted: yes ("Perlu kunjungan"). Referral to the puskesmas: **no**. Stock check: yes; the locker had stock and the package was reserved after approval with a pickup code. Drone dispatch: **not in the app** (no drone feature) |
| High risk, pregnant woman (4.9.3) | Second trimester; MUAC 22.8 cm; Hb 10.4 g/dL; two antenatal contacts | High | Midwife alerted; supplementary food and iron tablets reserved in the locker; follow-up scheduled | **High** ("Risiko tinggi", as expected): KEK (MUAC < 23.5 cm) and anaemia (Hb < 11 g/dL); the mother sees "contact the midwife" with Puskesmas Oesapa's number. Midwife alerted: **no midwife account in the app**; the village Kader was notified ("Ibu hamil perlu perhatian"). Supplementary food and iron tablets reserved in the locker: **no**; no supply request was created, and the mother's daily checklist asks her to take iron tablets and supplementary food. Follow-up scheduled: yes, next antenatal visit K3 due by 18 Dec 2026 |

# Summary

- **Expected level reached in 2 of 4 cases:** moderate-risk child and high-risk pregnant woman.
- **Low-risk girl rated medium.** With the family details unknown, the model is cautious: P(stunting) 0.31 against a
  medium threshold of 0.28. The result is flagged for review. Answering the family questions would decide the level.
- **High-risk child rated medium.** The app's WHO/Kemenkes guardrails raise a child to high only for severe
  stunting or wasting (z < −3), MUAC < 11.5 cm, oedema or danger signs. Stunting at −2.4 with 2T sets a minimum of
  medium. Either the expected level in Table 4.15 or the guardrail rule needs to change. That is a clinical decision.
- **Steps in the expected paths that the app does not perform:**
  - no referral to the puskesmas for the high-risk child;
  - no drone dispatch (not built);
  - no midwife account or alert;
  - no locker reservation of supplementary food and iron tablets for pregnant women.
- **Steps the expected paths do not mention:**
  - every supply request waits for an officer's approval before it is reserved;
  - the mother sees the pickup code (staff do not);
  - the low-risk girl also received a supply request.

# Detail per case

## Low risk (4.9.1)

| Item | Observed |
|---|---|
| Measurement | 18.0 months, 79.3 cm, 10.13 kg; HAZ −0.48, WAZ −0.08, WHZ +0.20 |
| Weighings | N, N, N (steady gain) |
| Model | ifls5-lr-2: P(stunting) 0.312 → medium; no guardrail |
| Final level | Medium; needs review (key inputs unknown) |
| Reasons shown | "Data belum lengkap (tinggi ayah, tinggi ibu, pendidikan ibu, daftar aset rumah tangga). Hasil ini perlu dicek petugas." |
| Triage | Kader home visit within 7 days; nutrition package at the nearest locker with a pickup code; weekly nutrition plan |
| Case | Opened, priority medium, assigned to the village Kader |
| Supply | PMT biscuits, Taburia, moringa powder → pending officer approval → after approval ready for pickup at Loker Posyandu Oesapa; mother sees a 6-digit code and QR (valid 3 days) |
| Notifications | Mother: "Hasil pemeriksaan", then "Paket gizi siap diambil". Kader and officer: "Perlu kunjungan: Wf Lowa" |

## Moderate risk (4.9.2)

| Item | Observed |
|---|---|
| Measurement | 10.0 months, 69.6 cm, 8.00 kg; HAZ −1.61, WAZ −1.24, WHZ −0.50 |
| Weighings | N, N, T (no gain at the last weighing) |
| Symptoms | "Anak diare sudah 3 hari minggu ini" → diarrhoea recognised |
| Model | P(stunting) 0.399 → medium; no guardrail |
| Final level | Medium; needs review (key inputs unknown) |
| Triage | Kader home visit within 7 days; nutrition package with pickup code; ORS with every loose stool and zinc once a day for 10 days; weekly nutrition plan |
| Case | Opened, priority medium |
| Supply | ORS + zinc, PMT biscuits, Taburia, moringa powder → after approval ready at Loker Posyandu Oesapa with pickup code and QR |
| Notifications | Mother: "Hasil pemeriksaan", "Paket gizi siap diambil". Kader and officer: "Perlu kunjungan: Wf Moda" |

## High risk, child (4.9.3)

| Item | Observed |
|---|---|
| Measurement | 26.0 months, 79.5 cm, 8.67 kg; HAZ −2.38, WAZ −2.66, WHZ −1.79 |
| Weighings | N, T, T (2T) |
| Model | P(stunting) 0.322 → medium |
| Guardrails | Stunted (HAZ < −2) and 2T → at least medium. Already medium, so no change; nothing raises it to high |
| Final level | Medium; needs review (key inputs unknown) |
| Reasons shown | Stunted (TB/U −2,4), low weight-for-age (BB/U −2,7), 2T, incomplete data |
| Triage | Kader home visit within 7 days; nutrition package with pickup code; weekly nutrition plan. No referral |
| Case | Opened, priority medium |
| Supply | PMT biscuits, moringa powder → after approval ready at Loker Posyandu Oesapa with pickup code and QR (locker had stock) |
| Notifications | Mother: "Hasil pemeriksaan", "Paket gizi siap diambil". Kader and officer: "Perlu kunjungan: Wf Higha" |

## High risk, pregnant woman (4.9.3)

| Item | Observed |
|---|---|
| Entered | 20 weeks (second trimester), ANC visits K1 and K2, MUAC 22.8 cm, Hb 10.4 g/dL |
| Flags | KEK (action), anaemia (action) |
| Level | "Risiko tinggi"; reasons "Lengan kecil (KEK)", "Kurang darah (anemia)"; contact advised |
| Facility shown | Puskesmas Oesapa, 0380-000-101, 1.2 km; ambulance 119 |
| Follow-up | Next ANC: K3, due now, target 18 Dec 2026; daily checklist: iron tablet, supplementary food, K3 |
| Notifications | Kader: "Ibu hamil perlu perhatian". Mother: none. Officer: none about this pregnancy |
| Supply | None created |
