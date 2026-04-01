# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 21214 resampled segments, 820730 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.9088) against `vq_ctw` at `K=512` (0.9923).
- `2.3`: compare Markov baselines `order 1` (0.8595) and `order 2` (0.8480) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 30 min, K=256, code 142: taxi, mean speed 5.77 km/h, mean step 2.89 km, radius 5.93 km.
- Interval 30 min, K=1024, code 760: taxi, mean speed 5.77 km/h, mean step 2.89 km, radius 5.93 km.
- Interval 30 min, K=512, code 28: taxi, mean speed 5.77 km/h, mean step 2.89 km, radius 5.93 km.
- Interval 30 min, K=1024, code 336: taxi, mean speed 8.40 km/h, mean step 4.20 km, radius 5.24 km.
- Interval 30 min, K=512, code 99: taxi, mean speed 7.58 km/h, mean step 3.79 km, radius 5.21 km.
- Interval 30 min, K=256, code 78: taxi, mean speed 8.88 km/h, mean step 4.44 km, radius 5.21 km.
- Interval 30 min, K=256, code 160: taxi, mean speed 6.37 km/h, mean step 3.19 km, radius 5.51 km.
- Interval 30 min, K=256, code 117: taxi, mean speed 9.21 km/h, mean step 4.61 km, radius 5.94 km.
- Interval 30 min, K=512, code 470: taxi, mean speed 25.97 km/h, mean step 12.98 km, radius 6.34 km.
- Interval 30 min, K=1024, code 566: taxi, mean speed 15.45 km/h, mean step 7.72 km, radius 7.60 km.
