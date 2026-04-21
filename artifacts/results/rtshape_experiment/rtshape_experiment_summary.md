# RT-shape experiment summary

## Configuration
- Branch: 20-29 matched
- Supervision: response_labels
- Epochs: 10
- Scales: 0.1, 0.3, 0.5
- lambda_rt = 1.0
- lambda_choice = 3.0
- lambda_cong = 0.3
- choice_temperature = 0.10
- time_steps_factor = 1.75
- rt_shape_focus = True

## Key observed result so far
- Scale 0.1, epoch 05: score = 0.3386, rt_score = 0.1615, model_accuracy = 0.9979, human_accuracy = 0.9660, model_congruency_gap = 0.2016, human_congruency_gap = 0.0585
- Scale 0.1, epoch 10: score = 0.3528, rt_score = 0.1859, model_accuracy = 0.9997, human_accuracy = 0.9660, model_congruency_gap = 0.2273, human_congruency_gap = 0.0585
- Scale 0.1 finished with best_score = 0.3528, pred_mean = 0.831s

## Interpretation
This experiment did not solve the RT distribution shape problem. Increasing the time horizon and emphasizing RT-shape-focused checkpoint selection did not make the model more human-like. Instead, the model still drifted toward near-ceiling ground-truth accuracy while exaggerating the congruency RT gap and preserving poor RT-shape fit.

## Conclusion
The core RT-distribution issue is not explained by a too-short time horizon alone. The next intervention should target the objective / decision regime more directly rather than only relaxing the RT ceiling.
