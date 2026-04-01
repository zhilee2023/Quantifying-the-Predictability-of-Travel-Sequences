# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 5 users, 5 resampled segments, 9197 samples.
- Modes observed: taxi.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.8766) against `vq_ctw` at `K=256` (0.8717).
- `2.3`: compare Markov baselines `order 1` (0.7558) and `order 2` (0.7084) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 15 min, K=256, code 124: taxi, mean speed 4.59 km/h, mean step 1.15 km, radius 5.13 km.
- Interval 15 min, K=256, code 222: taxi, mean speed 1.04 km/h, mean step 0.26 km, radius 3.96 km.
- Interval 15 min, K=256, code 166: taxi, mean speed 5.55 km/h, mean step 1.39 km, radius 5.03 km.
- Interval 15 min, K=256, code 229: taxi, mean speed 10.34 km/h, mean step 2.58 km, radius 6.47 km.
- Interval 15 min, K=256, code 147: taxi, mean speed 6.04 km/h, mean step 1.51 km, radius 4.73 km.
- Interval 15 min, K=256, code 34: taxi, mean speed 16.80 km/h, mean step 4.20 km, radius 6.39 km.
- Interval 15 min, K=256, code 9: taxi, mean speed 15.29 km/h, mean step 3.82 km, radius 6.56 km.
- Interval 15 min, K=256, code 44: taxi, mean speed 9.99 km/h, mean step 2.50 km, radius 6.40 km.
- Interval 15 min, K=256, code 31: taxi, mean speed 8.04 km/h, mean step 2.01 km, radius 5.13 km.
- Interval 15 min, K=256, code 140: taxi, mean speed 12.86 km/h, mean step 3.22 km, radius 6.70 km.
