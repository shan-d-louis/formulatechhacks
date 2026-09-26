# SIDEWALL: independent Spa dataset findings

## Decision
Use this recording to build and demonstrate braking-event review. It does not establish a tyre-failure predictor, safe stint limit, or reliable remaining-life model. Five laps from one simulator car are not enough to support those claims.

## Source and reproduction
- Dataset: https://huggingface.co/datasets/Nasim435/spa-francorchamps-lap-data
- MIT license; Assetto Corsa Chevrolet Corvette C7.R at Spa. This is simulator telemetry, not F1 telemetry.
- Dataset card describes five laps, but lap_progress is zero throughout, so this analysis cannot independently segment or verify those laps.
- CSV SHA-256: 6eadc93f64c9fab08a181b86ac644a1a936777ea3abdd754229ae67f9f57c697
- Run `python analyze_spa.py /path/to/spa_telemetry.csv` with pandas and numpy installed. Outputs go beside the script.

## Measured findings
- 51,554 rows and 84 channels covering 803.287 seconds (13 min 23 sec).
- Average sampling rate: 64.18 Hz. Median interval: 15.559 ms. Maximum interval: 30.892 ms. No non-increasing timestamps or gaps above 100 ms.
- No missing cells. This does not imply that every channel contains useful measurements.
- 13 constant channels, including lap_progress, all four brake temperatures, ambient and road temperatures, ABS, TC and brake bias. These cannot support learning responses to changing settings or weather within this session.
- Vehicle speed spans 1.10–252.85 km/h. Fuel spans 13.57–29.94 in source units.
- Core tyre temperatures span about 26.5–95.8 °C across wheels; pressure spans about 16.08–27.17 psi. These are observed simulator values, not safety thresholds.
- Wear values remain between 99.43 and 100. This provides very little degradation variation and no demonstrated end-of-life outcome.
- 92 braking episodes using brake > 0.2, speed > 50 km/h, continuous duration >= 0.2 s. These are threshold-defined episodes, not necessarily distinct corners or driver braking manoeuvres.
- Five candidate per-wheel slowdown events: FR at 157.925 s, FR at 506.796 s, RR at 789.142 s, FR at 790.532 s, FR at 790.843 s. Nearby detections can belong to the same manoeuvre.

## Detection method and limits
The source slip_* fields are nonnegative and sometimes exceed 100. Do not feed them into a signed slip-ratio threshold without verifying their definition.

Instead, fit an effective rolling radius for each wheel using 1,860 samples with speed > 80 km/h, brake < 0.01, throttle < 0.15 and |lateral acceleration| < 0.2 g. The fitted factors are approximately 0.346 m for the fronts and 0.357 m for the rears, assuming the wheel-speed values are radians/second.

Calculate proxy = wheel_speed × fitted_radius / vehicle_speed − 1. Flag brake > 0.2, speed > 50 km/h and proxy < −0.2 continuously for at least 80 ms. A timestamp gap above 100 ms breaks an episode.

This approximation uses vehicle speed rather than local speed at each wheel, ignores changing rolling radius and steering geometry, and is fitted and applied on the same session. It is exploratory event discovery, not independent validation. No video or independent labels were available in this analysis. Precision, recall and prediction lead time are unknown.

## Illustrative slowdown window, confounded by unloading
Show 504.8–509.0 seconds. The front-right candidate lasts 0.219 seconds, begins at 171.8 km/h, reaches full brake input, and has a minimum wheel-speed proxy of −0.948. Show speed, braking and all four wheel traces together. Label it “possible lock-up, review required”.

## Priority next steps
1. Use the supplied event CSV to build event selection and replay in the dashboard.
2. Review candidate windows and matched normal braking windows. If no replay video exists, collect controlled simulator sessions with normal braking and deliberately induced lock-ups, plus video.
3. Vary assists, temperature, fuel and tyres across new sessions. Record metadata and wheel-speed units explicitly.
4. Keep complete sessions separated for training and evaluation. A random row split leaks neighbouring telemetry and inflates results.
5. Start with the transparent rule baseline. Train ML only after defining independent labels and an evaluation set. A model reproducing its own rule-generated labels does not prove earlier detection.
6. Keep pressure/temperature as measured simulator context. Do not infer real F1 punctures, failure probabilities, or guaranteed safe laps from this dataset.

## Team message
I independently ran the Spa Corvette simulator dataset: 51,554 rows, 84 channels, 13m23s, actual average 64.18 Hz. I found 92 threshold-defined braking episodes and five possible wheel-slowdown events. An illustrative slowdown is the front-right event around 506.8 seconds at 172 km/h. Lap progress and brake temperatures are constant, and tyre wear barely changes, so this is useful for braking-event review but cannot establish remaining tyre life. I have the event timestamps, channel summary and reproducible script ready for integration.

## Verification and integration update
The analysis was rerun after adding input validation. A separate sample-by-sample event counter independently reproduced 92 braking episodes and all five candidate timestamps. Exported window IDs and dataset dimensions also passed checks.

Threshold sensitivity, with brake > 0.2 and speed > 50 km/h:
- Proxy < −0.10: 147 events at 80 ms minimum duration; 72 at 150 ms.
- Proxy < −0.20: 5 events at 80 ms; 2 at 150 ms.
- Proxy < −0.30: 4 events at 80 ms; 2 at 150 ms.

This sensitivity means the count depends strongly on the definition. It is not an independently established count of lock-ups. Keep the threshold visible in the dashboard and do not report accuracy before independent labeling.

`event_windows.csv` contains full-rate samples from two seconds before to two seconds after each candidate, with event_id, elapsed_s, input signals, per-wheel angular speed, core temperature, pressure and calculated slip proxies. Overlapping windows intentionally repeat samples under different event IDs; deduplicate timestamps before using them as a training dataset.

The repo track file spells the AI sponsor “Ampire”. This branch contains data analysis and review assets; it does not yet contain a trained AI model or a working pit-alert system.


## Important correction after tyre-load review
All five original candidate windows include zero reported load on the affected tyre:
- FR at 157.925 s: 6 of 8 samples.
- FR at 506.796 s: 8 of 15 samples.
- RR at 789.142 s: 8 of 14 samples.
- FR at 790.532 s: 3 of 7 samples.
- FR at 790.843 s: 4 of 10 samples.

These signals are confounded by wheel unloading or potentially unreliable load telemetry. A slowing unloaded wheel does not establish a tyre sliding on the track. The earlier proposed best demo at 506.8 s is therefore not a clean lock-up example.

Adding reported load > 0 to every qualifying sample, with the same 80 ms duration and other thresholds, leaves one candidate interval. Thresholds of >100 and >500 in source load units also leave one interval. This is a diagnostic sensitivity check, not validation of contact or a confirmed lock-up. See positive_load_candidates.csv. The source load signal itself still needs validation.

The raw slowdown count of five remains arithmetically correct. Present it as five rule-flagged wheel slowdowns, all with unloading confounds. Do not claim five verified lock-ups. event_windows.csv now includes all four load channels.

Rear-left and rear-right wheel speeds are exactly equal in 72.85% of samples. This could reflect simulator/drivetrain behavior or logging behavior; investigate before treating those channels as independent evidence.
