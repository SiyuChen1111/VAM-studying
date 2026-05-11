# RT Readout Smoke Summary

- Baseline run: `artifacts/results/rt_model_variational_ww_synthesis/smoke`
- DMC+Var→WW run: `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt`
- Baseline checkpoint: unknown
- DMC+Var→WW checkpoint: unknown

## Baseline vs DMC+Var→WW

- Mean RT: baseline=4.5229, DMC+Var→WW=0.4551, human=0.5964
- Median RT: baseline=4.9900, DMC+Var→WW=0.4414, human=0.5900
- Predicted skewness: baseline=-1.9348, DMC+Var→WW=0.8787, human=-0.3427
- Tail spread q95-q50: baseline=0.0000, DMC+Var→WW=0.2231, human=0.1447
- Error minus correct RT: baseline=0.7075, DMC+Var→WW=-0.0060, human=-0.0560
- Congruency gap: baseline=0.4953, DMC+Var→WW=0.0574, human=0.0246
- Response agreement: baseline=0.4844, DMC+Var→WW=0.7314

## Eval subset diagnostics

- Baseline subset mode: unknown
- DMC+Var→WW subset mode: unknown
- Baseline human-error trials: n/a
- DMC+Var→WW human-error trials: n/a
- Baseline congruent / incongruent: n/a / n/a
- DMC+Var→WW congruent / incongruent: n/a / n/a
- Baseline balance constraints satisfied: n/a
- DMC+Var→WW balance constraints satisfied: n/a

## Hard gates

- Mean RT stays near baseline scale: False
- Median RT stays near baseline scale: False
- Lower tail does not collapse earlier: False
- RT-scale gate passed: False

## Decision

- More right-skewed than baseline: True
- Error RT shifts later: False
- Congruency gap remains sensible: True
- Response agreement does not materially collapse: True
- Baseline error_minus_correct_rt defined: True
- DMC+Var→WW error_minus_correct_rt defined: True
- Baseline and DMC+Var→WW selected the same checkpoint: True
- Ranking tradeoff visible (baseline): False
- Ranking tradeoff visible (DMC+Var→WW): False
- Ranking tradeoff visible overall: False
- Worth scaling up: False
- Conclusion: reject and stay baseline
