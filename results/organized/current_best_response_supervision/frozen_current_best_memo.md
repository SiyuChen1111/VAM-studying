# Frozen current-best response-supervision memo

## Why this snapshot exists
The response-supervision training runs were stopped deliberately before full completion so that the current best-so-far checkpoints could be treated as this phase's working endpoint.

## Frozen best-so-far checkpoints

### 20-29 matched
- checkpoint = scale 0.2, epoch 5
- score = 0.6363
- rt_score = 0.2263
- model accuracy = 0.9993
- human accuracy = 0.9660
- model congruency RT gap = 0.0595
- human congruency RT gap = 0.0585
- model mean RT = 0.920 s
- human mean RT = 0.618 s

### 80-89
- checkpoint = scale 0.1, epoch 10
- score = 0.6489
- rt_score = 0.3406
- model accuracy = 1.0000
- human accuracy = 0.9791
- model congruency RT gap = 0.1054
- human congruency RT gap = 0.0854
- model mean RT = 0.599 s
- human mean RT = 0.939 s

## Interpretation
Both frozen best-so-far checkpoints occur at scale 0.2, suggesting a stable preferred regime under response-label supervision. Relative to the earlier target-supervision runs, the response-supervision branch initially moved model choice behavior closer to human behavior, but the best-so-far checkpoints still drifted back toward ceiling-level accuracy. The most reliable improvement is that the congruency RT gap became much more human-like, especially in the 80-89 branch.

## Parameter-comparison limitation
Because these runs were stopped before writing new best parameter files, this frozen snapshot supports behavioral comparison but does not yet support a clean parameter-level comparison under response-label supervision. To do that rigorously, a future rerun should save best-so-far parameters whenever a new best checkpoint is found.
