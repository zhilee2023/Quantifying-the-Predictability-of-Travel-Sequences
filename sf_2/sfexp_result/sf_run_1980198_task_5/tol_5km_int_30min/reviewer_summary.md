# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 21214 resampled segments, 820730 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.8366) against `vq_ctw` at `K=512` (0.9020).
- `2.3`: compare Markov baselines `order 1` (0.6802) and `order 2` (0.6736) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 30 min, K=256, code 114: taxi, mean speed 5.60 km/h, mean step 2.80 km, radius 5.88 km.
- Interval 30 min, K=1024, code 433: taxi, mean speed 5.57 km/h, mean step 2.79 km, radius 5.88 km.
- Interval 30 min, K=512, code 335: taxi, mean speed 5.49 km/h, mean step 2.74 km, radius 5.86 km.
- Interval 30 min, K=256, code 116: taxi, mean speed 4.88 km/h, mean step 2.44 km, radius 4.55 km.
- Interval 30 min, K=512, code 324: taxi, mean speed 6.44 km/h, mean step 3.22 km, radius 4.31 km.
- Interval 30 min, K=256, code 150: taxi, mean speed 8.74 km/h, mean step 4.37 km, radius 7.23 km.
- Interval 30 min, K=256, code 88: taxi, mean speed 9.66 km/h, mean step 4.83 km, radius 4.96 km.
- Interval 30 min, K=512, code 301: taxi, mean speed 8.87 km/h, mean step 4.43 km, radius 7.27 km.
- Interval 30 min, K=1024, code 282: taxi, mean speed 7.57 km/h, mean step 3.78 km, radius 7.23 km.
- Interval 30 min, K=512, code 151: taxi, mean speed 5.04 km/h, mean step 2.52 km, radius 4.98 km.
