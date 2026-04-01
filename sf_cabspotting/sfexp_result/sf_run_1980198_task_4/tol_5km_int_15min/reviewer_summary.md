# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 38090 resampled segments, 1517674 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.8466) against `vq_ctw` at `K=512` (0.9947).
- `2.3`: compare Markov baselines `order 1` (0.7273) and `order 2` (0.7233) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 15 min, K=256, code 27: taxi, mean speed 7.14 km/h, mean step 1.79 km, radius 5.90 km.
- Interval 15 min, K=1024, code 84: taxi, mean speed 7.14 km/h, mean step 1.78 km, radius 5.90 km.
- Interval 15 min, K=512, code 496: taxi, mean speed 7.12 km/h, mean step 1.78 km, radius 5.85 km.
- Interval 15 min, K=512, code 196: taxi, mean speed 10.86 km/h, mean step 2.72 km, radius 3.88 km.
- Interval 15 min, K=1024, code 667: taxi, mean speed 10.98 km/h, mean step 2.75 km, radius 3.87 km.
- Interval 15 min, K=256, code 55: taxi, mean speed 10.45 km/h, mean step 2.61 km, radius 3.85 km.
- Interval 15 min, K=512, code 374: taxi, mean speed 4.69 km/h, mean step 1.17 km, radius 7.20 km.
- Interval 15 min, K=512, code 381: taxi, mean speed 12.27 km/h, mean step 3.07 km, radius 6.67 km.
- Interval 15 min, K=256, code 31: taxi, mean speed 14.15 km/h, mean step 3.54 km, radius 7.25 km.
- Interval 15 min, K=1024, code 169: taxi, mean speed 11.62 km/h, mean step 2.91 km, radius 7.30 km.
