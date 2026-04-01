# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 38090 resampled segments, 1517674 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.8483) against `vq_ctw` at `K=512` (0.8647).
- `2.3`: compare Markov baselines `order 1` (0.5398) and `order 2` (0.5412) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 15 min, K=512, code 208: taxi, mean speed 5.99 km/h, mean step 1.50 km, radius 5.84 km.
- Interval 15 min, K=1024, code 561: taxi, mean speed 5.45 km/h, mean step 1.36 km, radius 5.79 km.
- Interval 15 min, K=512, code 44: taxi, mean speed 4.99 km/h, mean step 1.25 km, radius 5.54 km.
- Interval 15 min, K=1024, code 12: taxi, mean speed 5.32 km/h, mean step 1.33 km, radius 5.63 km.
- Interval 15 min, K=256, code 187: taxi, mean speed 5.87 km/h, mean step 1.47 km, radius 5.81 km.
- Interval 15 min, K=256, code 79: taxi, mean speed 4.11 km/h, mean step 1.03 km, radius 5.62 km.
- Interval 15 min, K=1024, code 389: taxi, mean speed 7.56 km/h, mean step 1.89 km, radius 5.75 km.
- Interval 15 min, K=1024, code 258: taxi, mean speed 7.63 km/h, mean step 1.91 km, radius 5.87 km.
- Interval 15 min, K=256, code 71: taxi, mean speed 2.21 km/h, mean step 0.55 km, radius 5.38 km.
- Interval 15 min, K=1024, code 63: taxi, mean speed 6.13 km/h, mean step 1.53 km, radius 5.56 km.
