"""Iteratively extract 4 datasets from Kaggle and save them as CSV files."""
import subprocess

import pandas as pd

from utils import root_relative_path


OUTPUT_PATH = root_relative_path("outputs/kaggle_tyre_strategy_output.csv")
TARGETS = [
    {
        "slug": "navenkumar1998/formula-1-dataset-with-weather-and-tyre-features",
        "kind": "strategy_dataset",
        "note": "Tyre/weather strategy source compatible with this extractor.",
    },
    {
        "slug": "aadigupta1601/f1-strategy-dataset-pit-stop-prediction",
        "kind": "strategy_dataset",
        "note": "Potential pit-stop strategy source; Kaggle may return 403 without access.",
    },
    {
        "slug": "playground-series-s6e5",
        "kind": "competition",
        "note": "Competition data needs `kaggle competitions download`, not dataset download.",
    },
    {
        "slug": "oshomuralidaran/pitwall-analytics",
        "kind": "context_dataset",
        "note": "Pit-wall context data; not a tyre-stint dataset with Compound/StintLength.",
    },
]


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored Markdown table."""
    if df.empty:
        return "_No rows._"

    text_df = df.fillna("").astype(str)
    headers = list(text_df.columns)
    rows = text_df.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def output_row_count() -> int:
    """Return the current appended output row count."""
    if not OUTPUT_PATH.exists():
        return 0
    return len(pd.read_csv(OUTPUT_PATH))


def print_markdown_preview(status_rows: list[dict]) -> None:
    """Print a concise Markdown preview of the appended output dataset."""
    if status_rows:
        print("\n## Kaggle Target Status\n")
        print(markdown_table(pd.DataFrame(status_rows)))

    if not OUTPUT_PATH.exists():
        print("\n## Kaggle Tyre Strategy Output Preview\n\nNo output CSV was created.")
        return

    df = pd.read_csv(OUTPUT_PATH)
    print("\n## Kaggle Tyre Strategy Output Preview")
    print(f"\n- Output CSV: `{OUTPUT_PATH}`")
    print(f"- Total valid rows: {len(df)}")

    if "Dataset" in df.columns:
        dataset_counts = df["Dataset"].value_counts(dropna=False).rename_axis("Dataset").reset_index(name="Rows")
        print("\n### Rows by Dataset\n")
        print(markdown_table(dataset_counts))

    if "Compound" in df.columns:
        compound_counts = (
            df["Compound"].value_counts(dropna=False).rename_axis("Compound").reset_index(name="Rows")
        )
        print("\n### Rows by Compound\n")
        print(markdown_table(compound_counts))

    preview_cols = [
        col
        for col in ("Dataset", "Year", "Event", "Session", "Driver", "Stint", "Compound", "StintLength", "AirTemp", "TrackTemp")
        if col in df.columns
    ]
    preview = df[preview_cols].head(10)
    print("\n### Data Preview\n")
    print(markdown_table(preview))


def classify_failure(details: str) -> str:
    """Convert extractor output into a concise failure reason."""
    if "Authentication required to call the Kaggle API" in details:
        return "Kaggle CLI is not authenticated in this environment."
    if "403 Client Error: Forbidden" in details:
        return "Kaggle returned 403 Forbidden; account lacks access or dataset is unavailable."
    if "does not include a recognizable compound column" in details:
        return "Downloaded files are not compatible with tyre-strategy schema."
    if "Kaggle dataset download failed" in details:
        return "Kaggle dataset download failed; verify slug, auth, and access."
    return "Extraction command failed; inspect child output for details."


def run_extraction(target: dict) -> dict:
    """Run the extraction for a given Kaggle dataset."""
    kaggle_dataset = target["slug"]
    if target["kind"] != "strategy_dataset":
        return {
            "Target": kaggle_dataset,
            "Kind": target["kind"],
            "Status": "skipped",
            "RowsBefore": output_row_count(),
            "RowsAfter": output_row_count(),
            "Reason": target["note"],
        }

    download_name = kaggle_dataset.replace("/", "__")
    # Construct the command to run the extraction script
    command = [
        "uv", "run", "python", str(root_relative_path("datasets/extract_kaggle_tyre_strategy.py")),
        "--kaggle-dataset", kaggle_dataset,
        "--download-dir", str(root_relative_path(f"data/raw/kaggle_tyre_strategy/{download_name}")),
        "--out", str(OUTPUT_PATH),
    ]
    
    # Execute the command
    before = output_row_count()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    after = output_row_count()
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        if details:
            tail = "\n".join(details.splitlines()[-6:])
            print(tail)
        reason = classify_failure(details)
        print(
            f"FAILED {kaggle_dataset} with exit code {result.returncode}. "
            "Continuing with the next dataset."
        )
        print(f"Total appended output rows still: {after}")
        return {
            "Target": kaggle_dataset,
            "Kind": target["kind"],
            "Status": "failed",
            "RowsBefore": before,
            "RowsAfter": after,
            "Reason": reason,
        }

    if result.stdout.strip():
        print(result.stdout.strip())
    print(f"SUCCESS {kaggle_dataset}: appended output rows {before} -> {after}")
    return {
        "Target": kaggle_dataset,
        "Kind": target["kind"],
        "Status": "success",
        "RowsBefore": before,
        "RowsAfter": after,
        "Reason": target["note"],
    }

def main():
    status_rows = []
    for target in TARGETS:
        result = run_extraction(target)
        status_rows.append(result)
        if result["Status"] == "skipped":
            print(f"SKIPPED {result['Target']}: {result['Reason']}")

    successes = sum(row["Status"] == "success" for row in status_rows)
    attempted = sum(row["Status"] != "skipped" for row in status_rows)
    print(
        f"Finished Kaggle extraction loop: {successes}/{attempted} attempted "
        f"targets succeeded; total output rows={output_row_count()}"
    )
    print_markdown_preview(status_rows)

if __name__ == "__main__":
    main()
