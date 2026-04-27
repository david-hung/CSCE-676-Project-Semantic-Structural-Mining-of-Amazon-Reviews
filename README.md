# CSCE 676 Final Project: Amazon Video Games Review Mining

This repository contains the final curated notebook and supporting code for a CSCE 676 project on the Amazon Review Data 2018 `Video_Games_5` category.

Start here: `main_notebook.ipynb`

Project video: https://www.youtube.com/watch?v=EonSS0aVjBI

## Project Question

The project studies one focused question:

**Do products that look similar in review language also occupy similar positions in a shared-reviewer graph?**

Short answer: **not strongly**. The final analysis finds weak alignment between semantic clusters and graph clusters, which suggests that review language and reviewer-overlap structure capture different aspects of the category.

The semantic view comes from TF-IDF representations of product-level review text. The structural view comes from a graph where two products are connected when the same reviewers appear on both products. The final notebook compares those two views and uses topic modeling to interpret the text clusters.

## Repository Layout

- `main_notebook.ipynb`: final notebook written as the project story
- `checkpoints/checkpoint_1.ipynb`: project checkpoint 1 notebook
- `checkpoints/checkpoint_2.ipynb`: project checkpoint 2 notebook
- `src/amazon_video_games_project/analysis.py`: reusable analysis pipeline
- `scripts/download_video_games_data.py`: helper script to fetch the Video Games 5-core file
- `scripts/run_video_games_analysis.py`: CLI entrypoint that runs the full analysis and writes figures/tables
- `scripts/build_final_notebook.py`: regenerates the curated final notebook with concrete findings
- `requirements.txt`: Python package requirements for the project

```text
CSCE-676-Project-Semantic-Structural-Mining-of-Amazon-Reviews/
├── README.md
├── final_deliverable.pdf
├── main_notebook.ipynb
├── requirements.txt
├── checkpoints/
│   ├── checkpoint_1.ipynb
│   └── checkpoint_2.ipynb
├── scripts/
│   ├── download_video_games_data.py
│   ├── run_video_games_analysis.py
│   └── build_final_notebook.py
└── src/
    └── amazon_video_games_project/
        ├── __init__.py
        └── analysis.py
```

## Dataset

- Dataset: Amazon Review Data (2018), `Video_Games_5`
- Source: [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html)
- Direct file: `https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz`
- Expected local path: `data/raw/Video_Games_5.json.gz`
- Download helper: `python scripts/download_video_games_data.py`
- Preprocessing: combine `summary` and `reviewText`, compute review metadata, aggregate reviews to product-level documents, keep the top 400 products with at least 15 reviews, build TF-IDF text features, and build a shared-reviewer graph with a minimum of 2 shared reviewers per edge.

The raw dataset is not committed because of size.

## Setup

Create and activate a virtual environment, then install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `matplotlib` complains about a non-writable config directory in WSL, set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

## Environment

- Python: `3.12.3`
- Key dependencies:
  - `pandas==3.0.2`
  - `numpy==2.4.4`
  - `scikit-learn==1.8.0`
  - `scipy==1.17.1`
  - `networkx==3.6.1`
  - `matplotlib==3.10.8`
  - `seaborn==0.13.2`

The full pinned environment is listed in `requirements.txt`.

## Download the Dataset

```bash
python scripts/download_video_games_data.py
```

Or manually place the downloaded file at `data/raw/Video_Games_5.json.gz`.

## Run the Analysis

```bash
python scripts/run_video_games_analysis.py
```

The script writes:

- `outputs/video_games/summary.json`
- `outputs/video_games/product_results.csv`
- `outputs/video_games/text_cluster_terms.csv`
- `outputs/video_games/topic_terms.csv`
- `outputs/video_games/figures/*.png`

To regenerate the notebook after rerunning the analysis:

```bash
python scripts/build_final_notebook.py
```

## Open the Final Notebook

After the dataset is present, open:

- `main_notebook.ipynb`

The notebook is designed to be the clean final project narrative rather than a scratchpad. It covers motivation, research question, preprocessing choices, semantic analysis, graph analysis, results, and conclusions.

## Methods Used

Course-aligned methods:

- TF-IDF text mining
- KMeans clustering
- Graph construction from shared reviewers
- PageRank, clustering coefficient, and community detection

Extension method:

- LDA topic modeling for cluster interpretation

## Ethical Notes

- Amazon reviews are self-selected and may overrepresent strong positive or negative opinions.
- Popular products attract more reviews and therefore more graph connections.
- Results are reported only at aggregate product level.

## License

This repository is licensed under the MIT License. The Amazon review data remains subject to its original terms.
