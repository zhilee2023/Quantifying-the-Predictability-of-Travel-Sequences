# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 68436 resampled segments, 531945 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.9398) against `vq_ctw` at `K=512` (0.9997).
- `2.3`: compare Markov baselines `order 1` (0.8993) and `order 2` (0.8960) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 5 min, K=256, code 195: taxi, mean speed 12.01 km/h, mean step 1.00 km, radius 1.84 km.
- Interval 5 min, K=512, code 413: taxi, mean speed 12.00 km/h, mean step 1.00 km, radius 1.84 km.
- Interval 5 min, K=1024, code 90: taxi, mean speed 11.88 km/h, mean step 0.99 km, radius 1.82 km.
- Interval 5 min, K=1024, code 587: taxi, mean speed 13.19 km/h, mean step 1.10 km, radius 1.96 km.
- Interval 5 min, K=1024, code 360: taxi, mean speed 14.32 km/h, mean step 1.19 km, radius 2.30 km.
- Interval 5 min, K=1024, code 894: taxi, mean speed 4.51 km/h, mean step 0.38 km, radius 2.11 km.
- Interval 5 min, K=512, code 248: taxi, mean speed 14.62 km/h, mean step 1.22 km, radius 3.11 km.
- Interval 5 min, K=1024, code 963: taxi, mean speed 13.68 km/h, mean step 1.14 km, radius 2.06 km.
- Interval 5 min, K=512, code 214: taxi, mean speed 17.76 km/h, mean step 1.48 km, radius 4.10 km.
- Interval 5 min, K=512, code 379: taxi, mean speed 7.71 km/h, mean step 0.64 km, radius 3.21 km.
