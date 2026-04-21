# Current-stage conclusion memo

## Scope
This memo summarizes what can be concluded before the 20-29 Stage 2 run fully finishes.

## Current result status
- 80-89: formal Stage 2 result available
- 20-29: Stage 2 still running; best completed-scale interim result available

## 80-89 formal result
- Best scale: 0.200
- Best score: 0.6448
- Human mean RT: 0.838s
- Model mean RT: 0.432s
- Human accuracy: 0.9615
- Model accuracy: 1.0000
- Human congruency RT gap: 0.0830s
- Model congruency RT gap: 0.0914s

Interpretation: the model captures the approximate congruency RT gap and maintains very high accuracy, but remains far too fast overall and fails to reproduce the heavy right-skew expected in older-adult behavior.

## 20-29 interim result
- Best completed-scale interim scale: 0.100
- Best completed-scale interim score: 0.5753
- Human mean RT: 0.605s
- Model mean RT: 0.487s
- Human accuracy: 0.9343
- Model accuracy: 1.0000
- Human congruency RT gap: 0.0397s
- Model congruency RT gap: 0.0814s
- Live progress note: scale 4/5, epoch 01/20

Interpretation: the same broad pathology is already visible in the young group—accuracy is too close to ceiling and RT remains too fast—although the temporal mismatch appears less severe than in the old group.

## Cross-group interim pattern
Across both groups, the fitted model captures relative conflict structure more successfully than the absolute temporal regime of human decisions. The current Stage 2 solutions appear to live in a too-efficient decision regime: highly accurate, conflict-sensitive, but insufficiently slow and insufficiently heavy-tailed.

## Most important implication
The most valuable next analysis is not yet a strong age-mechanism claim, but a diagnosis of why the shared Stage 1 + current Stage 2 fitting setup systematically produces overly fast and overly accurate behavior.

## Immediate next step after 20-29 finishes
Run the full research-plan-aligned comparison in the required order:
1. behavior / human signatures
2. parameter comparison
3. mechanism / trajectory geometry

## Interim visual assets generated now
- Figure A1. 80-89 human vs model behavioral signatures
- Figure A2. 80-89 RT distribution comparison (x-axis constrained to 0-2 s)
- Figure A3. Interim 20-29 versus final 80-89 comparison
- Figure A4. Interim trajectory geometry using 20-29 current-best scale and 80-89 formal fit
