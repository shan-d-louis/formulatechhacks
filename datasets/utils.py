from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def root_relative_path(path: str) -> Path:
    out_path = Path(path)
    if out_path.is_absolute():
        return out_path
    return PROJECT_ROOT / out_path

def append_or_write(df: pd.DataFrame, out: str) -> None:
    out_path = root_relative_path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Appending to an existing file: {out_path}")
        existing_df = pd.read_csv(out_path)
        df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(
            subset=["Year", "Event", "Session", "Driver", "date"],
            keep="last",
        )
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(
        df.groupby(["Year", "Event", "Session", "Driver"])
        .size()
        .rename("Rows")
        .reset_index()
        .to_string(index=False)
    )