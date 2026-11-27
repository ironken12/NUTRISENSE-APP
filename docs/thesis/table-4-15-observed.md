---
title: "Table 4.15 Workflow Test Cases: revised table and observed results"
---

# How the tests were run

- **Date:** 27 November 2026.
- **App version:** risk model ifls5-lr-2, branch `claude/sharp-bohr-t1b7pz`, commit `f3a65aa`.
- **Script:** `backend/scripts/workflow_cases.py`, which reruns everything below.
- **Database:** a fresh demo database with the demo villages, lockers, Kader, officer and doctor accounts.
- **Accounts and data entry:** each case used a new mother account (consent given). The profile went in through the
  app's normal API, then the last measurement (or the mother's check-up values) triggered the system.
- **Measurements:** heights and weights were computed from the WHO 2006 tables so that the last measurement gives the
  profile's HAZ and WHZ. Earlier monthly weighings were set to give the profile's weight-gain pattern (N = gained
  enough, T = did not gain, by the Buku KIA minimum).
- **Officer approval:** supply requests wait for an officer's approval, which was given in the test, as in practice.
- **Database shortcut:** the app has no screen for entering a past pregnancy's antenatal visits. For the low-risk case,
  the mother's delivered pregnancy (six antenatal visits, her height and education) was therefore linked to the child
  directly in the database. Everything else went through the app.
- **Unspecified values:** values the table does not give were chosen as neutral: WHZ −0.5 for the moderate-risk boy;
  20 weeks, 152 cm and junior high school for the pregnant woman. The family details are unknown for cases 2 and 3,
  as for a newly registered family.

# Two runs

**First run (before the changes).** 2 of 4 cases reached the expected level:
- **Low-risk girl rated medium.** Her family details were not in the profile, so the model's P(stunting) was 0.31,
  just above the medium threshold. She was also sent a food package.
- **High-risk child rated medium.** No rule raised stunting (HAZ −2.4) with 2T to high, and there was no referral.
- **Pregnant woman:** no supplementary food or iron tablets were reserved.
- **Steps the app does not have:** drone dispatch and midwife alerts.

**Changes made after the first run.** These were decided with the author, as design decisions.

| Change | Where | Reason |
|---|---|---|
| Stunting (HAZ < −2) together with 2T or low weight-for-age (WAZ < −2) sets **high**, with a referral to the puskesmas | App (guardrail rule) | Each is a reason to refer a child from the Posyandu to the puskesmas (Kemenkes). **To be confirmed by the expert panel** |
| A food package only when the measurements show a growth problem (wasting, stunting, underweight, T or 2T, falling growth line) or the result is high | App (triage) | Supplementary food is limited and meant for children with a growth problem. A medium from background risk alone gets the Kader visit and advice |
| KEK → supplementary food for pregnant women (PMT ibu hamil); anaemia → iron tablets (TTD), reserved in a locker after officer approval | App (maternal check, supply requests) | Kemenkes programme for pregnant women; uses the same locker pathway as children's packages |
| The low-risk profile includes the family details | Table 4.15 | So the case tests a child whose data are complete. With the details unknown, the app deliberately rates the child medium and asks for review (Section B.6) |
| Drone dispatch removed | Table 4.15 | Not part of the app. Stock comes from the locker, or by road from a hub when the locker has none |
| "Midwife alerted" replaced by "Kader alerted; mother shown the puskesmas contact" | Table 4.15 | The app has no midwife account; a midwife role is future work |

**Second run (after the changes).** 4 of 4 cases reached the expected level and path (table below).

# Table 4.15 (revised) with the Observed column

| Case | Synthetic profile | Expected level | Expected system path | Observed |
|---|---|---|---|---|
| Low risk (4.9.1) | Girl, 18 months; HAZ −0.5, WHZ +0.2; steady weight gain; birth weight 3,100 g; mother had six antenatal contacts, is 157 cm and finished senior high school, and was 27 at the birth; father 168 cm; second child; one child under 5 at home; 7 of the 10 household assets | Low | Growth chart updated; age-appropriate feeding education; next routine visit scheduled; no supply request | **Low** ✓. HAZ −0.48, WHZ +0.20, weighings N-N-N; P(stunting) 0.20. "Pertumbuhan dalam batas normal" (growth within normal limits). Routine monthly Posyandu weighing and the weekly nutrition plan. No case, no supply request, no staff alert |
| Moderate risk (4.9.2) | Boy, 10 months; HAZ −1.6; no weight gain at the last weighing; birth weight 2,400 g; diarrhoea in the past month | Medium | Case opened for Kader follow-up; nutrition guidance; supplementary food reserved in the locker after officer approval; QR pickup code sent | **Medium** ✓. HAZ −1.61, weighings N-N-T; P(stunting) 0.40. Case opened for a Kader visit within 7 days. ORS and zinc for 10 days, and the weekly nutrition plan. ORS + zinc, PMT, Taburia and moringa powder: after approval, reserved at Loker Posyandu Oesapa; the mother received "Paket gizi siap diambil" with a 6-digit code and QR (valid 3 days). Flagged for review: family details unknown |
| High risk, child (4.9.3) | Girl, 26 months; HAZ −2.4, WHZ −1.8; no weight gain at two consecutive weighings | High | Case assigned to the Kader; referral to the puskesmas (doctor within 48 hours, Kader helps); officer and doctor alerted; supplementary food reserved in the locker after officer approval (from a hub by road if the locker has no stock) | **High** ✓. HAZ −2.38, WAZ −2.66, WHZ −1.79, weighings N-T-T (2T). The model gave medium (P 0.32); the stunted-with-faltering rule raised it to high. "Rujuk ke Puskesmas: periksakan anak ke dokter dalam 48 jam. Kader akan membantu rujukan." (refer to the puskesmas, doctor within 48 hours, the Kader helps). High-priority case assigned to the Kader. Kader, officer and doctor alerted ("Risiko tinggi"). PMT and moringa powder: after approval, reserved at Loker Posyandu Oesapa (stock available) with a pickup code and QR. Flagged for review: rule raised the level, family details unknown |
| High risk, pregnant woman (4.9.3) | Second trimester; MUAC 22.8 cm; Hb 10.4 g/dL; two antenatal contacts | High | Kader alerted; mother shown the puskesmas contact; supplementary food and iron tablets reserved in the locker after officer approval; next antenatal visit scheduled | **High** ✓ ("Risiko tinggi": KEK and anaemia). Kader alerted ("Ibu hamil perlu perhatian"); mother shown Puskesmas Oesapa, 0380-000-101, and ambulance 119. PMT ibu hamil and iron tablets requested; officer alerted ("Permintaan paket ibu hamil"). After approval, reserved at Loker Posyandu Oesapa; the mother received "Paket gizi siap diambil" with a code and QR. Next antenatal visit K3 due by 18 Dec 2026 |

# Notes for the thesis

- The stunted-with-faltering rule is a design decision based on Kemenkes referral practice. It should be confirmed by
  the expert panel (midwife and nutritionist) before field use.
- The workflow tests show that the system follows the designed paths for synthetic profiles. They are not a clinical
  evaluation.
- The high-risk child's path has no separate "home visit within 7 days" step. The case is assigned to the Kader, who
  helps with the puskesmas referral within 48 hours.
- Raw results of the second run: `workflow-cases-results.json`.
