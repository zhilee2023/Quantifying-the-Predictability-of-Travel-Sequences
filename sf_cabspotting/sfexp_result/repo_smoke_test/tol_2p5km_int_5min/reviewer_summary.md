# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 3 users, 3 resampled segments, 15504 samples.
- Modes observed: taxi.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.8984) against `vq_ctw` at `K=256` (0.9493).
- `2.3`: compare Markov baselines `order 1` (0.8622) and `order 2` (0.8298) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 5 min, K=256, code 17: taxi, mean speed 6.52 km/h, mean step 0.54 km, radius 5.55 km.
- Interval 5 min, K=256, code 132: taxi, mean speed 3.52 km/h, mean step 0.29 km, radius 4.00 km.
- Interval 5 min, K=256, code 76: taxi, mean speed 0.38 km/h, mean step 0.03 km, radius 3.63 km.
- Interval 5 min, K=256, code 103: taxi, mean speed 22.78 km/h, mean step 1.90 km, radius 6.78 km.
- Interval 5 min, K=256, code 14: taxi, mean speed 1.54 km/h, mean step 0.13 km, radius 4.44 km.
- Interval 5 min, K=256, code 36: taxi, mean speed 19.90 km/h, mean step 1.66 km, radius 7.12 km.
- Interval 5 min, K=256, code 135: taxi, mean speed 6.97 km/h, mean step 0.58 km, radius 6.01 km.
- Interval 5 min, K=256, code 137: taxi, mean speed 13.37 km/h, mean step 1.11 km, radius 6.09 km.
- Interval 5 min, K=256, code 189: taxi, mean speed 7.23 km/h, mean step 0.60 km, radius 5.35 km.
- Interval 5 min, K=256, code 70: taxi, mean speed 12.15 km/h, mean step 1.01 km, radius 7.13 km.
