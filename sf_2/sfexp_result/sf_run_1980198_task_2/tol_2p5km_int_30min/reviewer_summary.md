# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 21214 resampled segments, 820730 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.8458) against `vq_ctw` at `K=512` (0.8554).
- `2.3`: compare Markov baselines `order 1` (0.5054) and `order 2` (0.5064) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 30 min, K=1024, code 400: taxi, mean speed 4.52 km/h, mean step 2.26 km, radius 5.76 km.
- Interval 30 min, K=512, code 383: taxi, mean speed 3.61 km/h, mean step 1.81 km, radius 5.73 km.
- Interval 30 min, K=256, code 144: taxi, mean speed 2.92 km/h, mean step 1.46 km, radius 5.71 km.
- Interval 30 min, K=1024, code 63: taxi, mean speed 3.77 km/h, mean step 1.88 km, radius 5.56 km.
- Interval 30 min, K=512, code 76: taxi, mean speed 3.94 km/h, mean step 1.97 km, radius 5.69 km.
- Interval 30 min, K=1024, code 952: taxi, mean speed 4.78 km/h, mean step 2.39 km, radius 5.65 km.
- Interval 30 min, K=256, code 103: taxi, mean speed 4.05 km/h, mean step 2.03 km, radius 5.66 km.
- Interval 30 min, K=256, code 228: taxi, mean speed 4.48 km/h, mean step 2.24 km, radius 5.83 km.
- Interval 30 min, K=1024, code 326: taxi, mean speed 5.58 km/h, mean step 2.79 km, radius 5.81 km.
- Interval 30 min, K=1024, code 716: taxi, mean speed 11.22 km/h, mean step 5.61 km, radius 6.78 km.
