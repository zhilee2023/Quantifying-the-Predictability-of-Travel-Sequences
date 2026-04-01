# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 534 users, 68820 resampled segments, 3556130 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.8640) against `vq_ctw` at `K=512` (0.8663).
- `2.3`: compare Markov baselines `order 1` (0.8023) and `order 2` (0.7602) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 5 min, K=256, code 22: taxi, mean speed 6.04 km/h, mean step 0.50 km, radius 5.73 km.
- Interval 5 min, K=1024, code 45: taxi, mean speed 7.01 km/h, mean step 0.58 km, radius 5.76 km.
- Interval 5 min, K=512, code 48: taxi, mean speed 6.33 km/h, mean step 0.53 km, radius 5.75 km.
- Interval 5 min, K=1024, code 674: taxi, mean speed 4.90 km/h, mean step 0.41 km, radius 5.55 km.
- Interval 5 min, K=512, code 191: taxi, mean speed 4.96 km/h, mean step 0.41 km, radius 5.75 km.
- Interval 5 min, K=512, code 210: taxi, mean speed 4.58 km/h, mean step 0.38 km, radius 5.43 km.
- Interval 5 min, K=256, code 81: taxi, mean speed 4.79 km/h, mean step 0.40 km, radius 5.53 km.
- Interval 5 min, K=256, code 88: taxi, mean speed 8.94 km/h, mean step 0.75 km, radius 5.75 km.
- Interval 5 min, K=512, code 109: taxi, mean speed 7.79 km/h, mean step 0.65 km, radius 5.63 km.
- Interval 5 min, K=256, code 50: taxi, mean speed 8.66 km/h, mean step 0.72 km, radius 5.66 km.
