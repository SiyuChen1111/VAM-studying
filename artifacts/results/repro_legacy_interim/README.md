# Legacy interim reproduction bundle

This folder is the formal output bundle for the pinned legacy interim replay entrypoint:
`code/scripts/generate_legacy_interim_reproduction.py`.

## Primary figures/tables
- `figureA2_80_89_rt_distributions.png`
- `figureA4_interim_trajectory_geometry.png`
- `figureA4_interim_trajectory_spread.csv`

## Supporting files
- `figure_hybrid_legacy_parameter_comparison.png`
- `hybrid_legacy_parameter_comparison.csv`
- `hybrid_legacy_parameter_notes.md`
- `legacy_reference_comparison.md`
- `legacy_reference_image_comparison.csv`
- `legacy_reference_spread_comparison.csv`
- `legacy_reproduction_manifest.json`

## Intended use
- `figureA2_80_89_rt_distributions.png` = congruent/incongruent RT distribution replay in legacy context, with 80-89 on the top row and 20-29 on the bottom row. No RT mean alignment is applied; all four panels share the same plotting scale.
- `figureA4_interim_trajectory_geometry.png` = legacy hybrid trajectory geometry replay.
- `figureA4_interim_trajectory_spread.csv` = numeric A4 spread table for direct comparison with the legacy reference.
- `figure_hybrid_legacy_parameter_comparison.png` / `hybrid_legacy_parameter_comparison.csv` = internal parameter comparison for the user-requested hybrid setup, showing that noise-related terms are shared while scale differs.