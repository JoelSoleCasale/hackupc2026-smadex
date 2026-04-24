#!/usr/bin/env python3
"""
Execute all 6 EDA notebooks and compile them into a single HTML report
that shows only text and plot outputs (no source code).

Run from the smadex/ directory:
    uv run python eda/build_report.py
"""
import sys
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = [
    "01_dataset_overview.ipynb",
    "02_creative_performance.ipynb",
    "03_creative_fatigue.ipynb",
    "04_feature_importance.ipynb",
    "05_geo_os_analysis.ipynb",
    "06_creative_assets.ipynb",
]

EDA_DIR = Path(__file__).parent
OUTPUT_HTML = EDA_DIR / "eda_report.html"


def execute_notebook(nb_path: Path) -> nbformat.NotebookNode:
    print(f"  [{NOTEBOOKS.index(nb_path.name) + 1}/{len(NOTEBOOKS)}] {nb_path.name}", flush=True)
    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    # path tells the kernel where to set cwd — notebooks use ../ to load CSVs
    ep.preprocess(nb, {"metadata": {"path": str(EDA_DIR)}})
    return nb


def merge_notebooks(notebooks: list[nbformat.NotebookNode]) -> nbformat.NotebookNode:
    merged = nbformat.v4.new_notebook()
    merged.metadata = notebooks[0].metadata
    for nb in notebooks:
        merged.cells.extend(nb.cells)
    return merged


def build_html(nb: nbformat.NotebookNode, output_path: Path) -> None:
    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    output_path.write_text(body, encoding="utf-8")


def main() -> None:
    print("=== Smadex EDA Report Builder ===\n")

    print("Step 1/3  Executing notebooks...")
    try:
        executed = [execute_notebook(EDA_DIR / nb) for nb in NOTEBOOKS]
    except Exception as exc:
        print(f"\nExecution failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nStep 2/3  Merging notebooks...")
    merged = merge_notebooks(executed)
    print(f"  {len(merged.cells)} cells merged from {len(NOTEBOOKS)} notebooks.")

    print("\nStep 3/3  Generating HTML report...")
    build_html(merged, OUTPUT_HTML)
    print(f"  Saved: {OUTPUT_HTML.resolve()}")

    print("\nDone. Open eda/eda_report.html in your browser.")


if __name__ == "__main__":
    main()
