# RT-shape tail-loss experiment summary

## Configuration
- Branch: 20-29 matched
- Supervision: response_labels
- Epochs: 10
- Scales: 0.1, 0.3, 0.5
- lambda_rt = 1.0
- lambda_choice = 3.0
- lambda_cong = 0.3
- lambda_tail = 0.2
- choice_temperature = 0.10
- tail_quantiles = 0.90, 0.95, 0.99
- time_steps_factor = 1.0

## Best observed result in this experiment
- Scale 0.1, epoch 10
- score = 0.5722
- rt_score = 0.4941
- model_accuracy = 0.9955
- human_accuracy = 0.9660
- response_agreement = 0.9618
- model_congruency_rt_gap = 0.1140
- human_congruency_rt_gap = 0.0585
- pred_mean = 0.633 s

## Comparison with simpler Run B baseline
Reference Run B (without tail loss) performed better overall:
- score ≈ 0.6004
- response_agreement ≈ 0.9647
- pred_mean ≈ 0.595 s

## Conclusion
Adding a small quantile tail loss improved over the earlier pure time-horizon experiment, but it still did not outperform the simpler response-agreement-focused Run B baseline. Therefore, the current evidence does not support adopting this tail-loss configuration as the new default path.
