# Lightning Response

**Elevator pitch:** AI pit wall that predicts tyre failures, lock-ups and wheelspin from telemetry, explains the cause, and tells the driver and crew when to box. It called BOX 15 laps before a real F1 failure.

## Inspiration
Since 2022, every F1 car has carried a standard FIA tyre-pressure sensor, but there is no public analytics layer that predicts tyre failure. Tyre problems can develop faster than a pit crew can recognise them in scattered telemetry: a lock-up can flat-spot one tyre, overheat it, or leave its pressure drifting, and a tyre can fail before the team makes a pit call. Tyre failures have caused dangerous incidents, including:
- **Baku 2021:** Verstappen and Stroll had blowouts at around 300 km/h.
- **Silverstone 2020:** three front-left failures in the final laps.
- **Qatar 2023:** kerbs caused sidewall damage, and the FIA imposed an emergency 18-lap limit per set.
- **Nürburgring 2005:** a flat spot vibrated a suspension to failure.

Public F1 data has a second problem: it does not show each tyre's pressure, temperature or wheel speed, so a warning can look more certain than the evidence allows.

Lightning Response watches every tyre in real time. It detects lock-ups, wheelspin, overheating, cold tyres, flat spots and air loss, predicts how many laps each tyre has left, and alerts the driver and the pit crew when it's time to manage the tyres or box. It turns available telemetry and historical stint data into earlier per-tyre risk warnings, an estimate of safe laps remaining, and a clear recommendation, so safety risks are caught before they become critical, using data-driven, calibrated and explained AI models.

## What it does
- **Predicts** lock-ups and wheelspin about a second ahead, and detects overheating, cold tyres, flat spots and air loss.
- **Explains** every risk: a calibrated probability, the cause (braking, throttle, speed and cornering, engine and gearing, tyre heat, pressure), how much of the tyres' grip is in use, and a plain-language fix.
- **Warns the driver directly:** when lock-up or wheelspin risk is high, the driver's phone shows a warning with the action to take; when it is imminent, it says BRACE.
- **Forecasts tyre life:** laps until the performance cliff, with a "safe laps left" promise calibrated to hold 90% of the time.
- **Acts:** one health score (0–100) per tyre and one pit call, read out as a radio message.
- **Proves it on real races:** replays of Silverstone 2020 and Baku 2021 with models that never saw those seasons, including the moment each tyre really failed.
- **Lets judges drive:** a phone becomes the pedals of a live simulated car; four scenario buttons force a lock-up, wheelspin, a pressure leak or overheating so the pit wall can be seen catching each one.

## How we built it
### Addressing the main problems
| Initiative | What Lightning Response shows |
|---|---|
| **Safety diagnosis** | Real-time per-tyre health (0–100) and an escalating pit call. Lock-up and wheelspin warnings come 0.5–1.75 s early, and the safe-laps bound is ready before the cliff arrives. Silverstone 2020 is called BOX 15 laps before the real failure. |
| **Data-driven** | Six seasons of public F1 data (2018–2021, 2024–2025: 127 races, 4,990 stints, 138,683 laps) turned into a tyre-safety dataset, and a **data-driven stint limit for 33 circuits**: the Qatar rule, set before anything breaks. |
| **Reliable AI** | Calibrated, explained early-warning models trained on simulator ground truth and run on real F1 telemetry, a virtual tyre sensor, and survival models with a conformally calibrated safe-laps bound. |
| **End-to-end connection** | The driver's phone is the car's pedals and receives risk alerts, over any network through a Cloudflare tunnel. |

### How the prediction works
```
car data (4×/s) ─┬─> virtual tyre sensor ──────────> temperature + pressure per tyre
                 ├─> lock-up / wheelspin model ───> calibrated % + cause + advice
                 ├─> air-loss and flat-spot detectors (physics)
                 └─> lap data ──> tyre-life survival models ──> safe laps left
                                          │
              all of it ──> tyre health 0–100 ──> OK / MANAGE / BOX ──> banner, radio, phones
```

- **Lock-up and wheelspin:** LightGBM trained on 116 human stints in Assetto Corsa (a Dallara F317), where every wheel's slip is recorded, using only signals a public F1 feed also has. A second, per-car layer adds tyre temperature and pressure; TreeSHAP splits each prediction into its causes.
- **Virtual tyre sensor:** estimates each tyre's temperature from speed, pedals and position; pressure follows from the gas law.
- **Air loss and flat spots:** physics detectors. Air loss tracks the amount of gas in the tyre (pressure divided by temperature) with a CUSUM test, so heating alone never triggers it; flat spots combine locked-sliding distance with a once-per-revolution vibration.
- **Tyre life:** survival models on real races that ask, lap by lap, "given this tyre has lasted so far, how likely is the cliff on the next lap?", plus a conformally calibrated "safe laps left" promise.

### The pit wall
- **Banner:** the one call the crew needs (OK, MANAGE, BOX THIS LAP, BOX NOW), with its reasons and the radio message. Level 1 is called ADVISE in the engine and on the phones. It switches to **CRASH** or **TYRE FAILURE** when that happens.
- **Tyres:** a top-down drawing of the car with each tyre's tile beside its own wheel. Each wheel glows in its tyre's temperature colour (blue cold, green in the grip window, yellow hot, red overheating) and flashes when the tyre is flagged for a flat spot, air loss or deflation. Replays show *AI estimates*; the live car shows *tyre sensors*.
- **Track map:** the car (a hand-drawn sprite pointing along its direction of travel) on the real circuit, with a glow in the call colour, pins for lock-ups, wheelspin and raised calls, and a burst where a crash happened.
- **Risk, next second:** lock-up and wheelspin probability, grip in use, the causes with measured evidence, and advice. Each box is outlined by alert level:

  | Colour | When | Label |
  |---|---|---|
  | Red, pulsing | live: BRACE sent to the driver; replay: the event is happening, or risk ≥ 30% | BRACE · SENT TO DRIVER / HAPPENING / HIGH RISK |
  | Red | live: a warning sent to the driver | WARNING · SENT TO DRIVER |
  | Orange | a pit stop is called (BOX THIS LAP or BOX NOW) | PIT STOP CALLED |
  | Yellow | risk ≥ 10%, or raised over the last few corners | WATCH |

  In live mode, a strip above the boxes shows exactly what the driver's phone is showing.
- **Tyre life:** safe laps left (90% confidence) and the likely laps to the cliff.
- **Log:** every call change, event, driver alert, scenario result and crash, with time stamps.

### Replays: real races the models never saw
| Replay | Real outcome | What the pit wall does |
|---|---|---|
| British GP 2020, Hamilton, laps 20–52 | Front-left failure on the last lap; limped home on three wheels and won | BOX THIS LAP from lap 37, held to the failure on lap 52. At the failure moment the banner shows **TYRE FAILURE** and the car carries on at reduced speed, as in the telemetry. |
| Azerbaijan GP 2021, Verstappen, laps 14–46 | Left-rear failure at over 300 km/h on lap 46; crashed out of the lead | BOX by lap 28. At the failure moment the banner shows **CRASH** and the car stops, as in the telemetry. The call names the front-right tyre, not the left-rear: the real cause (running pressure) is not visible in public data. |

- Each replay uses tyre-life models **retrained without that race's season**, so the race is genuinely unseen.
- Everything on screen uses only data available up to that moment.
- The failure moment is found in the telemetry itself: the first point on the failure lap where the car is at least 30% slower than on the previous lap at the same place for 3 s. If it then stops within 10 s, it is shown as a crash; otherwise as a tyre failure.
- Key-moment buttons jump to the first MANAGE, the first BOX and the real failure.

**Why BOX came 15 laps early at Silverstone 2020.** Hamilton pitted on lap 13 under the safety car and ran 39 laps on one set of hards. By lap 37 the set was 24 laps old, where Silverstone sets normally reach the performance cliff at about 26.5 laps, and his lap times had started drifting slower than the stint's own trend (0.19 s above it on the latest lap). The model put the chance of a cliff within 3 laps at 5.9% and the tyre-health score fell below the BOX threshold. If his lap times had still looked like lap 26, that chance would have been 2.3%. The model predicted the end of the tyres' useful life, not the puncture itself; that slow build-up is the same pattern behind all three late front-left failures that day.

### Drive it yourself: the live simulator
The live car runs on the Silverstone racing line from a real F1 lap, with a physics tyre model: 3 thermal nodes per tyre (tread surface, carcass, inflation gas), grip that depends on temperature, pressure, wear and downforce, pressure from the gas law, and the sensors a real car carries (infrared tread, TPMS, wheel speed, and a hub accelerometer for flat-spot vibration). Physics runs at 20 Hz; the full monitoring stack analyses it 4 times a second, as soon as each new data frame exists.

#### Scenario buttons
The phone's four scenario buttons drive the car through shared scripts (`simulator/scenarios.json`, also used by the browser simulator and the backend tests). The script takes the pedals; a script's `steer` becomes a virtual corner. Tap the lit button again to stop early; afterwards the phone keeps the wheel.

| Button | Script | What happens | Result seen in testing |
|---|---|---|---|
| Lock-up | `lockup` | Three braking zones, each later; the last stamps on the brakes | Warned about 4 s early on the near-limit zone; lock-up caught within 0.2 s |
| Wheelspin | `wheelspin` | Three hairpin exits, each harder; the last floors it from a standstill | Warned on the approach; wheelspin caught within 0.2 s |
| Tyre Overheating | `corner` | Fast corners, then a long tight one past the limit | Warned about 0.2 s before the tread passed 125 °C |
| Tyre Pressure Anomaly | `puncture` | Debris cuts a random tyre, which leaks while the car keeps racing | Air loss detected about 1.6 s after the cut |

When a scenario ends, a watcher times the pit wall's warning against the simulator's own record of when the hazard happened, and the phone and the pit-wall log show, e.g., *"Lightning Response warned 0.2 s before the overheating. It also warned 1 time on the approach."* The lead time is measured from the warning that runs into the hazard, so an earlier unrelated warning can't inflate it; a miss is reported as a miss.

#### Driver alerts
The driver's phone is warned directly from the next-second risk:

| Level | Lock-up | Wheelspin | Phone shows |
|---|---|---|---|
| Warning (counteract) | risk ≥ 30% | risk ≥ 80% | amber "⚠ LOCK-UP RISK / Ease off the brake" (or throttle), short buzz |
| Brace (imminent, or happening) | risk ≥ 50%, or a lock-up is detected | risk ≥ 95%, or wheelspin is detected | red, pulsing "‼ … IMMINENT / BRACE", red screen border, three long buzzes |

An alert stays up for at least 1 s and reaches the phone about 0.3–0.4 s after the data that triggers it.

#### Reacting to crashes
The car crashes and stops dead when it goes **off the track** (too fast for a corner, beyond the slide limit) or a **tyre fails** (a tyre that has lost 40% of its air, above 80 km/h). The pit wall shows CRASH with the reason and a countdown, the phone shows CRASHED and vibrates, a running scenario stops, and after 8 s the car is recovered to the pits on fresh tyres. Tidy driving and the scenarios never crash.

## Challenges we ran into
- **Phones on any network.** Phones normally reach the laptop over the local Wi-Fi, which fails on networks that block device-to-device traffic (eduroam, most venue Wi-Fi) and breaks whenever the laptop reconnects. A Cloudflare quick tunnel gives the server a public `https://…trycloudflare.com` address instead, and the QR code on screen always points to it.
- **No tyre sensors in public data.** FastF1 has speed, pedals and position, but no tyre temperature, pressure or wheel speed. We had to build a virtual tyre sensor and train the early warnings on simulator data where every wheel's slip is recorded, using only signals a public F1 feed also has.
- **Getting warnings to arrive in time.** A 0.5 s warning is useless if it reaches the driver 0.5 s late. Lining the analysis up with each new data frame cut alert delay from about 0.56 s to 0.43 s.

## Accomplishments that we're proud of
- **Warnings that beat human reaction time.** On drivers our models never trained on, lock-up warnings come a median 0.5 s early and wheelspin warnings 1.75 s early: 2× to 7× a driver's roughly 0.25 s reaction time. We catch 85% of lock-ups and 99% of wheelspins before they happen.
- **Calling a real F1 tyre failure 15 laps early.** On a replay of Silverstone 2020, a season the model never saw, Lightning Response called BOX on lap 37; Hamilton's front-left failed on lap 52. We can explain exactly why: the tyres were past Silverstone's normal life, and his lap times had started to slip.
- **Probabilities you can trust.** When the model says 20% risk, it happens about 1 time in 5 (calibration error 0.1–0.2%). Our "safe laps left" promise held 93% of the time on the unseen 2025 season, against a 90% target.
- **Seeing what public data can't.** Our virtual tyre sensor estimates tyre temperature within 6–9 °C on a track it never saw, 40–50% better than a naive estimate, and pressure within 0.98 psi.
- **Beating overfitting.** Our first tyre-life model memorised races; we cut the gap between training and test scores from 0.15 to 0.05 while keeping test accuracy.
- **A complete, live system.** A judge's phone becomes the pedals, over any network; risk updates 4 times a second, BRACE alerts reach the driver in 0.3–0.4 s, and crashes stop the car. It's backed by 287 automated tests.

## What we learned
- **Random data splits lie.** Our first model scored 0.997 on training data but 0.85 on held-out seasons; weather columns alone identified individual races. We now split by season, use only information available at that point in the race, and retrain the replay models without the season being replayed.
- **For safety, calibration matters as much as accuracy.** A pit call is only useful if "20%" really means 20%, so we calibrated every probability and checked it on unseen data.
- **Simulator data can teach a real-world model, if you're strict about it.** Training on simulator ground truth while only allowing signals a public F1 feed has is what lets the model run on real telemetry.
- **Test the simulator before blaming the model.** Our wheelspin alerts fired constantly during "tidy" driving. The cause was the simulator's autopilot, which really did spin the rear tyres within a second of 13% of moments. Fixing it cut false wheelspin alerts from 11 to 0 per 150 seconds.
- **Be precise about what a model predicts.** At Silverstone our model predicted the end of the tyres' useful life, not the puncture itself; saying that clearly makes the result more credible, not less.

## What's next for Lightning Response
- **Real sensor data.** Every F1 car already carries tyre-pressure sensors, and teams have tread-temperature and wheel-speed sensors too. Plugging these in would replace our estimates with measurements.
- **Better failure prediction.** Public data has only 60 tyre failures (15 official), so our failure model is our weakest (0.68). Team or tyre-supplier data would let us predict *which* tyre will fail, not just when a set is worn out.
- **Fewer false alarms.** Lock-up warnings still fire about 4 times per 150 s of normal driving in our simulator; we want to calibrate them on real car data.
- **More races and series.** We skipped 2022–23 to save download time, and Qatar's circuit isn't in our data yet. After that: F2, F3, GT and karting, where teams have fewer engineers watching the data.
- **A permanent deployment.** Moving from a laptop and a quick tunnel to an always-on server, so crews and drivers can use it at any track.

## Built with
Python, FastAPI, Uvicorn, WebSockets, LightGBM, scikit-learn, pandas, NumPy, SciPy, PyArrow, joblib, FastF1, OpenF1, Jolpica/Ergast, Assetto Corsa Gym (Hugging Face), Kaggle, HTML, CSS, JavaScript, Canvas 2D, SVG, qrcode, Cloudflare Tunnel, pytest, uv, Git, GitHub.
