# Integrated current-results analysis

## Behavioral patterns from the human data
The human behavior figures show the classic signatures emphasized in `research_proposal_v4.md`: age-related slowing, skewed RT distributions, congruency effects, and an error-slower component. The `80-89` group is slower on average than the matched `20-29` group, and the RT-distribution panels make the age difference visually obvious rather than reducing it to a single mean.

## Frozen current-best model comparison
Under the current frozen best-so-far response-supervision checkpoints, both model branches still drift back toward ceiling-level accuracy. However, the congruency RT gap is much closer to human behavior than it was under target-label supervision. This suggests that switching to response supervision corrected part of the regime problem, but did not eliminate the model's tendency toward overly idealized choice behavior.

## How to read Figure A4 right now
`figureA4_interim_trajectory_geometry.png` is still useful, but it should be interpreted as a geometry preview from the earlier supervision path rather than the final response-supervision mechanism figure. The spread summary shows:

| age_group   | condition   |   mean_state_space_spread |
|:------------|:------------|--------------------------:|
| 20-29       | Congruent   |                  0.231621 |
| 20-29       | Incongruent |                  0.165734 |
| 80-89       | Congruent   |                  0.481482 |
| 80-89       | Incongruent |                  0.450729 |

This older geometry preview suggests that the `80-89` branch occupies a much broader state-space regime than `20-29`, regardless of congruency condition. That pattern is directionally compatible with the broader research hypothesis that older adults may require a noisier or more variable decision-dynamics regime. But because the response-supervision runs did not write best-so-far parameter snapshots before we stopped them, this geometry result cannot yet be treated as the final mechanism analysis for the corrected supervision branch.

## What is solid now
1. Human-side behavior plots can already support age-related slowing, skewness, congruency, and error-slower claims.
2. Frozen response-supervision model summaries can support a cautious behavioral comparison.
3. The old geometry figure can be used as a provisional mechanism preview, not as the final mechanism result.

## What remains blocked
True response-supervision model RT distributions and updated trajectory geometry still require saved best-so-far parameter files or full trial-level predictions from the corrected branch.
