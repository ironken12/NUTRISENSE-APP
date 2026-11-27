# Risk model: retraining results on IFLS5

Data: IFLS5 (RAND, 2014–15), 4,794 children aged 0–59 months with a measured height, 4,299 households, 13 provinces
(no NTT). Built with `build_ifls5.py`. The IFLS data is not in this repository (RAND licence).

Target: stunting, HAZ < −2 (WHO standards, the app's own `growth.py`): 35.6% of the children.
Evaluation: the same two settings as the thesis, a household-grouped random 80/20 split and NTB (province 52) held out.
Survey weights: IFLS5 cross-section health-measurement weight (`pwt14usxa`). Capacity: Kader visit the top 30%.

Check of the extraction: on this table the thesis model (ifls5-lr-1) has AUC 0.692 for stunting (thesis: 0.681), and
the input means match the model's own scaling means (age 30.4 vs 29.5 months, birth weight 3,140 vs 3,134 g,
ANC 9.3 vs 9.2, education 2.44 vs 2.46). The thesis reports 5,202 children; this extraction keeps 4,794 (children
with a measured height, a birth date and |HAZ| ≤ 6).

## Models

| Model | Inputs |
|---|---|
| Thesis model (ifls5-lr-1) | The 16 thesis inputs, scored exactly as the app does |
| New logistic regression | Same inputs, no district dummies, age spline, missing-value indicators, regularisation by household CV, survey weights |
| New LR + extra inputs | Plus father's height, mother's age at birth, birth order, children under 5 in the household, asset score (0–10) |
| Gradient boosting | Same inputs as the previous row |
| Rule | Mother < 150 cm or birth weight < 2.5 kg |

The extra inputs were chosen before any results were seen (from the model review), not by trying inputs until the
test score went up.

## Discrimination (AUC, 95% CI from 300 household resamples)

| Model | Random split (n = 975) | NTB held out (n = 456) |
|---|---|---|
| Thesis model (ifls5-lr-1) | 0.641 (0.596–0.681) | 0.674 (0.611–0.719) |
| New LR, thesis inputs | 0.639 (0.597–0.681) | 0.697 (0.642–0.745) |
| **New LR + extra inputs** | **0.662 (0.619–0.707)** | **0.719 (0.668–0.766)** |
| Gradient boosting + extra inputs | 0.655 (0.615–0.695) | 0.723 (0.672–0.765) |
| Rule | 0.581 (0.541–0.622) | 0.574 (0.517–0.622) |

The thesis model was trained on the random split of IFLS5, so some of these test children were in its training data
(all of NTB was available to it). Its figures here are optimistic; the new models' are not.

## Paired comparisons (same children, 1,000 household resamples; `compare.py`)

| Comparison | Random split | NTB held out |
|---|---|---|
| Thesis model − rule | **+0.061** (+0.025 to +0.097) | **+0.100** (+0.060 to +0.141) |
| New LR + extra − rule | **+0.082** (+0.049 to +0.117) | **+0.145** (+0.107 to +0.186) |
| New LR + extra − new LR, thesis inputs | **+0.023** (+0.006 to +0.040) | **+0.023** (+0.002 to +0.046) |
| New LR + extra − thesis model | +0.021 (−0.006 to +0.045) | **+0.045** (+0.016 to +0.074) |
| New LR, thesis inputs − thesis model | −0.003 (−0.023 to +0.015) | +0.022 (−0.002 to +0.047) |

Bold: the 95% CI excludes zero.

## At Kader capacity (top 30% visited)

| Model | Random split: recall / precision | NTB held out: recall / precision |
|---|---|---|
| Thesis model | 42.3% / 49.3% | 42.4% / 68.9% |
| New LR + extra inputs | 41.3% / 48.1% | 45.1% / 73.8% |

Visiting 30% of children finds about 4 in 10 stunted children, at about 1.4–2 visits per stunted child found.

![NTB held out](results/curves-ntb-held-out.png)

## What this supports

1. A model adds value over a simple checklist: both the thesis model and the new model rank children significantly
   better than "mother < 150 cm or birth weight < 2.5 kg", in both settings.
2. Inputs a Kader can collect at registration (father's height, mother's age, birth order, children under 5, a short
   asset checklist) improve the model significantly in both settings.
3. The new model generalises better to an unseen province (NTB): +0.045 AUC over the thesis model, which had seen NTB.
4. Gradient boosting is not better than logistic regression, so the interpretable model is the right choice.

## Limits (state these in the thesis)

- Discrimination is modest (AUC 0.66–0.72). Background factors alone cannot identify stunting precisely; the model
  is for prioritising visits, not for diagnosis. The WHO z-score rules on measured growth stay in charge.
- All models under-predict risk in NTB (calibration plot). NTT, with more stunting, will need recalibration on NTT
  children (`recalibrate.py`) before the probabilities or threshold are used there.
- In the decision curve the models beat "visit everyone" only above a threshold probability of about 0.35: they help
  when Kader cannot visit everyone, which is the case they are built for.
- Cross-sectional: the target is current stunting. Predicting future growth faltering needs repeated measurements.
- Immunisation comes only from the vaccination card in IFLS5 (78% unknown), so its effect is poorly estimated.
- IFLS5 is from 2014–15 and does not include NTT. Not a clinical validation.

## Reproduce

```bash
python ml/build_ifls5.py --dta /path/to/hh14_all_dta --out children.csv
python ml/train.py --data children.csv --holdout-province 52 --weight-col weight --capacity 0.30 --bootstrap 300 --out ml/out
python ml/compare.py --data children.csv --base-drop-extras --holdout-province 52 --weight-col weight
```
