# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 38090 resampled segments, 1517674 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.8384) against `vq_ctw` at `K=512` (1.0000).
- `2.3`: compare Markov baselines `order 1` (0.7350) and `order 2` (0.7401) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 15 min, K=256, code 117: taxi, mean speed 7.30 km/h, mean step 1.82 km, radius 5.93 km.
- Interval 15 min, K=512, code 365: taxi, mean speed 7.30 km/h, mean step 1.82 km, radius 5.93 km.
- Interval 15 min, K=1024, code 707: taxi, mean speed 7.30 km/h, mean step 1.82 km, radius 5.93 km.
- Interval 15 min, K=256, code 72: taxi, mean speed 11.32 km/h, mean step 2.83 km, radius 3.91 km.
- Interval 15 min, K=512, code 109: taxi, mean speed 11.32 km/h, mean step 2.83 km, radius 3.91 km.
- Interval 15 min, K=1024, code 468: taxi, mean speed 11.32 km/h, mean step 2.83 km, radius 3.91 km.
