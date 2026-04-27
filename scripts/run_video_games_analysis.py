"""CLI entrypoint for the Amazon Video Games final project analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amazon_video_games_project.analysis import AnalysisConfig, run_analysis


def main() -> None:
    review_path = PROJECT_ROOT / "data/raw/Video_Games_5.json.gz"
    output_dir = PROJECT_ROOT / "outputs/video_games"
    if not review_path.exists():
        raise FileNotFoundError(
            "Expected the dataset at data/raw/Video_Games_5.json.gz. "
            "Run scripts/download_video_games_data.py first."
        )

    results = run_analysis(review_path, output_dir, AnalysisConfig())
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved tables and figures under {output_dir}")


if __name__ == "__main__":
    main()
