# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 21214 resampled segments, 820730 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.8237) against `vq_ctw` at `K=512` (1.0000).
- `2.3`: compare Markov baselines `order 1` (0.6900) and `order 2` (0.6981) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 30 min, K=256, code 208: taxi, mean speed 5.77 km/h, mean step 2.89 km, radius 5.93 km.
- Interval 30 min, K=512, code 82: taxi, mean speed 5.77 km/h, mean step 2.89 km, radius 5.93 km.
- Interval 30 min, K=1024, code 952: taxi, mean speed 5.77 km/h, mean step 2.89 km, radius 5.93 km.
- Interval 30 min, K=512, code 298: taxi, mean speed 8.64 km/h, mean step 4.32 km, radius 5.29 km.
- Interval 30 min, K=1024, code 98: taxi, mean speed 8.64 km/h, mean step 4.32 km, radius 5.29 km.
- Interval 30 min, K=256, code 199: taxi, mean speed 8.63 km/h, mean step 4.32 km, radius 5.29 km.
- Interval 30 min, K=256, code 136: taxi, mean speed 28.93 km/h, mean step 14.46 km, radius 8.41 km.
- Interval 30 min, K=256, code 140: taxi, mean speed 12.39 km/h, mean step 6.20 km, radius 5.49 km.
- Interval 30 min, K=256, code 176: taxi, mean speed 3.18 km/h, mean step 1.59 km, radius 7.57 km.
