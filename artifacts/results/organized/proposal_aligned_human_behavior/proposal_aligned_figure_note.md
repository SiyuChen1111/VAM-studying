# Proposal-aligned behavior figure note

Generated figures:
- Figure P1. Human RT distributions by age, congruency, and correctness
- Figure P2. Human behavioral signatures by age group
- Figure P3. Frozen current-best model summary by age group

Why these figures are available now:
- Human-side RT distributions, skewness, congruency effects, and error-slower summaries can be computed directly from the age-group test CSVs.
- Frozen current-best model summaries can be computed from the saved comparison CSV extracted from response-supervision logs.

What is not yet available:
- True model RT distribution plots under response-label supervision for both age groups
- Updated response-supervision trajectory geometry analogous to Figure 4

These model-level distribution and geometry plots require saved best-so-far parameter files or trial-level model predictions, which were not written before the runs were stopped.
