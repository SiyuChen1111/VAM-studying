# Nondeterminism audit

## Classification table

| Source | Classification | Evidence | Why it matters |
|---|---|---|---|
| Subset sampling in sweep scripts | confirmed deterministic | `run_20_29_fixed_time2_readout_sweep.py:207-224` uses `np.random.default_rng(args.seed)` and fixed subset sizes | Subset membership should reproduce under the same cached inputs and seed. |
| Training DataLoader order | confirmed nondeterministic | `train_age_groups_efficient.py:709-710` creates `DataLoader(..., shuffle=True)` with no fixed generator | Batch order can change between nominally identical runs. |
| WW dynamics during training | confirmed nondeterministic | `vgg_wongwang_lim.py:289-312` and `337-358` call `torch.randn(...)` inside recurrent dynamics | Training trajectories and losses vary unless torch RNG is controlled. |
| Evaluation/inference simulation | confirmed nondeterministic | `compute_stage2_outputs` in `train_age_groups_efficient.py:399-417` performs two WW simulations, both stochastic | Even evaluation of the same params can drift because fresh noise is sampled. |
| Urgency readout function itself | confirmed deterministic | `vgg_wongwang_lim.py:125-188` has no randomness once `evidence_traj` is fixed | The urgency readout is not the source; the trajectory generation is. |
| Deterministic backend flags | suspected nondeterministic | No visible `torch.use_deterministic_algorithms(True)` or backend determinism flags in the inspected path | Backend kernels may drift across devices or runs. |
| Device/backend selection | suspected nondeterministic | `train_age_groups_efficient.py:1613-1619` auto-selects MPS/CUDA/CPU | Different backends can change floating-point behavior. |
| Checkpoint selection key | suspected nondeterministic | `train_age_groups_efficient.py:543-599, 869-889` uses float tuple ranking with a hard gate on `model_congruency_rt_gap <= 0.065` | Small metric noise can change which checkpoint is declared best. |
| Metric provenance mismatch | confirmed nondeterministic-for-comparison | `train_age_groups_efficient.py:849-889` evaluates/stores best results on training data, while `1333-1353` computes fresh test predictions later | Stored and recheck metrics can disagree even without config drift because they are not the same evaluation target. |
| Post-analysis urgency semantics | confirmed nondeterministic-for-comparison | `run_matched_full_age_group_analysis.py:107-120` and `run_age_group_post_analysis.py:96-109` use `model.ww.inference` then `decision_times_class.min(dim=1)` | Those paths can bypass urgency readout semantics entirely, making urgency rechecks not apples-to-apples. |

## Answers to the key questions

1. **Is subset sampling reproducibly seeded?** Yes, for the sweep scripts inspected.
2. **Is the trainer fully deterministic after subset selection?** No. Training uses shuffled batches and stochastic WW noise without an explicit torch-determinism policy.
3. **Is evaluation fully deterministic?** No. Evaluation/inference also resimulates noisy WW dynamics.
4. **Can best-checkpoint selection drift even when total score changes only slightly?** Yes. The ranking key uses a float tuple plus a hard congruency-gap gate.
5. **Are there implicit backend differences that could matter?** Yes. Auto device selection and missing deterministic backend flags leave that possibility open.

## Most likely root causes

- Unseeded stochastic WW simulation in both training and inference.
- Stored urgency metrics are training-eval metrics, while later reconciliations often rely on fresh test-time predictions.
- Some downstream post-analysis code reconstructs baseline-style RT from `decision_times_class.min(...)` instead of respecting the saved urgency readout configuration.