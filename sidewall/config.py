from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
FASTF1_CACHE = CACHE / "fastf1"
WEIGHTS = ROOT / "models" / "weights"

# FastF1 race-level output
LAPS_DIR = PROCESSED / "laps"          # one parquet per race: lap-level timing + tyre + weather
RC_DIR = PROCESSED / "race_control"    # one parquet per race: race-control messages
RESULTS_DIR = PROCESSED / "results"    # one parquet per race: classification + retirement status
TELEM_DIR = PROCESSED / "telemetry"    # one parquet per race: car + position data (all drivers)

THULAB_PARQUET = RAW / "thulab_spa.parquet"

SEASONS = list(range(2018, 2026))

for _d in (RAW, PROCESSED, FASTF1_CACHE, WEIGHTS, LAPS_DIR, RC_DIR, RESULTS_DIR, TELEM_DIR):
    _d.mkdir(parents=True, exist_ok=True)
