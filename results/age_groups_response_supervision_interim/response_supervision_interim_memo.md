# Response-supervision interim memo

## Current best-so-far checkpoints from `partial_best`

### 20-29 matched
- checkpoint = response_supervision_best_scale0.1_epoch20
- score = 0.6699
- rt_score = 0.4869
- model accuracy = 0.9868
- human accuracy = 0.9660
- model congruency RT gap = 0.0854
- human congruency RT gap = 0.0585
- model mean RT = 0.640 s
- human mean RT = 0.618 s

### 80-89
- checkpoint = response_supervision_best_scale0.1_epoch20
- score = 0.6044
- rt_score = 0.3932
- model accuracy = 1.0000
- human accuracy = 0.9791
- model congruency RT gap = 0.1344
- human congruency RT gap = 0.0854
- model mean RT = 0.706 s
- human mean RT = 0.939 s

## Main interpretation
These outputs are now based on the saved `partial_best` checkpoints rather than stale log parsing. At the current best-so-far points, both age groups still show near-ceiling model accuracy, but the congruency RT gap is much closer to the human target than under the original target-label supervision. The remaining mismatch is now easier to localize: the model captures conflict structure better than the full human temporal regime.
