# Cabspotting / San Francisco Reviewer Summary

- Dataset: Cabspotting / San Francisco, 533 users, 68436 resampled segments, 531945 samples.
- Modes observed: taxi.

- User/trajectory/label aggregates and the summary statistics below use **eligible segments only** (`eligible_for_vq`, same minimum length as VQ); see `eligible_segment_coverage.json` per scenario.

## Reviewer mapping

- `1.8`: compare `direct_ctw` (0.9976) against `vq_ctw` at `K=512` (1.0000).
- `2.3`: compare Markov baselines `order 1` (0.9963) and `order 2` (0.9961) against VQ-VAE.
- `2.5`: compare `linear` and `nearest` interpolation rows in `segment_level_results.csv` and `user_level_results.csv`.
- `2.6`: inspect `latent_code_summary.csv` for dominant modes, speed, step length, and radius-of-gyration profiles.
- `2.7`: Dataset-specific run metadata is captured in `run_metadata.json` and the segment/user-level CSV outputs.
- `3.3`: use `predictability_by_interval.png` and `codebook_sensitivity.png` for interval and codebook-size sensitivity.

## Most common latent primitives

- Interval 5 min, K=256, code 51: taxi, mean speed 12.01 km/h, mean step 1.00 km, radius 1.84 km.
- Interval 5 min, K=512, code 134: taxi, mean speed 12.01 km/h, mean step 1.00 km, radius 1.84 km.
- Interval 5 min, K=1024, code 237: taxi, mean speed 12.01 km/h, mean step 1.00 km, radius 1.84 km.
