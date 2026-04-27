"""Download the Amazon Video Games 5-core dataset used in the final project."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz"
OUTPUT_PATH = Path("data/raw/Video_Games_5.json.gz")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL} -> {OUTPUT_PATH}")
    urlretrieve(DATA_URL, OUTPUT_PATH)
    print("Download complete.")


if __name__ == "__main__":
    main()
