# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 38090 resampled segments, 1517674 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.9418) against `vq_ctw` at `K=512` (1.0000).
- `2.3`: compare Markov baselines `order 1` (0.9139) and `order 2` (0.9073) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 15 min, K=256, code 78: taxi, mean speed 7.30 km/h, mean step 1.82 km, radius 5.93 km.
- Interval 15 min, K=512, code 211: taxi, mean speed 7.30 km/h, mean step 1.82 km, radius 5.93 km.
- Interval 15 min, K=1024, code 240: taxi, mean speed 7.30 km/h, mean step 1.82 km, radius 5.93 km.
- Interval 15 min, K=256, code 148: taxi, mean speed 11.32 km/h, mean step 2.83 km, radius 3.91 km.
- Interval 15 min, K=512, code 241: taxi, mean speed 11.32 km/h, mean step 2.83 km, radius 3.91 km.
- Interval 15 min, K=1024, code 322: taxi, mean speed 11.31 km/h, mean step 2.83 km, radius 3.91 km.
- Interval 15 min, K=1024, code 517: taxi, mean speed 14.80 km/h, mean step 3.70 km, radius 3.61 km.
- Interval 15 min, K=1024, code 264: taxi, mean speed 75.57 km/h, mean step 18.89 km, radius 4.91 km.
- Interval 15 min, K=1024, code 189: taxi, mean speed 50.92 km/h, mean step 12.73 km, radius 4.29 km.
- Interval 15 min, K=1024, code 270: taxi, mean speed 71.79 km/h, mean step 17.95 km, radius 5.91 km.
