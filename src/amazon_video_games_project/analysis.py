"""End-to-end analysis for the Amazon Video Games final project."""

from __future__ import annotations

import ast
import gzip
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

RANDOM_STATE = 42


@dataclass(slots=True)
class AnalysisConfig:
    """Settings for the Video Games semantic-versus-structural analysis."""

    top_products: int = 400
    min_reviews_per_product: int = 15
    min_shared_reviewers: int = 2
    max_tfidf_features: int = 4000
    max_topic_features: int = 2500
    cluster_candidates: tuple[int, ...] = field(default_factory=lambda: (3, 4, 5, 6, 7, 8))
    num_topics: int = 6
    top_terms_per_group: int = 10
    top_products_in_plot: int = 15


def _parse_review_line(line: str) -> dict[str, Any]:
    line = line.strip()
    if not line:
        return {}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def load_reviews(review_path: str | Path) -> pd.DataFrame:
    """Load the gzipped Video Games review file into a tidy DataFrame."""

    rows: list[dict[str, Any]] = []
    with gzip.open(review_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = _parse_review_line(line)
            rows.append(
                {
                    "reviewer_id": record.get("reviewerID", ""),
                    "asin": record.get("asin", ""),
                    "overall": float(record.get("overall", 0.0) or 0.0),
                    "review_text": str(record.get("reviewText", "") or ""),
                    "summary": str(record.get("summary", "") or ""),
                    "unix_review_time": int(record.get("unixReviewTime", 0) or 0),
                }
            )

    reviews = pd.DataFrame(rows)
    reviews["combined_text"] = (
        reviews["summary"].fillna("").str.strip() + ". " + reviews["review_text"].fillna("").str.strip()
    ).str.strip(". ")
    reviews["review_length_words"] = reviews["combined_text"].str.split().str.len().fillna(0).astype(int)
    reviews["review_year"] = pd.to_datetime(
        reviews["unix_review_time"], unit="s", errors="coerce"
    ).dt.year.fillna(0).astype(int)
    return reviews


def prepare_product_documents(reviews: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the most-reviewed products and aggregate their text."""

    base_product_stats = (
        reviews.groupby("asin")
        .agg(
            review_count=("asin", "size"),
            mean_rating=("overall", "mean"),
            unique_reviewers=("reviewer_id", "nunique"),
            median_review_length=("review_length_words", "median"),
        )
        .sort_values(["review_count", "unique_reviewers"], ascending=[False, False])
    )

    selected_asins = (
        base_product_stats.loc[base_product_stats["review_count"] >= config.min_reviews_per_product]
        .head(config.top_products)
        .index
    )
    selected_reviews = reviews.loc[reviews["asin"].isin(selected_asins)].copy()

    product_documents = (
        selected_reviews.groupby("asin")
        .agg(
            review_count=("asin", "size"),
            mean_rating=("overall", "mean"),
            unique_reviewers=("reviewer_id", "nunique"),
            median_review_length=("review_length_words", "median"),
            first_review_year=("review_year", "min"),
            last_review_year=("review_year", "max"),
            document=("combined_text", lambda values: " ".join(text for text in values if text)),
        )
        .sort_values(["review_count", "unique_reviewers"], ascending=[False, False])
        .reset_index()
    )
    return selected_reviews, product_documents


def select_best_k(matrix: Any, candidates: tuple[int, ...]) -> tuple[int, float]:
    """Pick the cluster count with the highest silhouette score."""

    best_k = 2
    best_score = float("-inf")
    n_samples = matrix.shape[0]

    for candidate in candidates:
        if candidate >= n_samples:
            continue
        model = KMeans(n_clusters=candidate, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_k = candidate
            best_score = score

    if best_score == float("-inf"):
        return 2, float("nan")
    return best_k, best_score


def extract_top_terms(
    model: KMeans,
    feature_names: np.ndarray,
    top_terms_per_group: int,
) -> list[dict[str, Any]]:
    """Summarize each KMeans cluster with its highest-weight terms."""

    summaries: list[dict[str, Any]] = []
    for cluster_id, center in enumerate(model.cluster_centers_):
        top_indices = center.argsort()[::-1][:top_terms_per_group]
        summaries.append(
            {
                "cluster": cluster_id,
                "top_terms": ", ".join(feature_names[index] for index in top_indices),
            }
        )
    return summaries


def run_text_analysis(
    product_documents: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float], sparse.spmatrix]:
    """Build TF-IDF product vectors, choose k, and cluster the products."""

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=config.max_tfidf_features,
        min_df=3,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(product_documents["document"])
    best_k, best_silhouette = select_best_k(tfidf_matrix, config.cluster_candidates)

    cluster_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    product_documents["text_cluster"] = cluster_model.fit_predict(tfidf_matrix)

    text_svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    coordinates = text_svd.fit_transform(tfidf_matrix)
    product_documents["text_x"] = coordinates[:, 0]
    product_documents["text_y"] = coordinates[:, 1]

    cluster_terms = extract_top_terms(
        cluster_model,
        vectorizer.get_feature_names_out(),
        config.top_terms_per_group,
    )
    diagnostics = {
        "best_text_k": int(best_k),
        "text_silhouette": float(best_silhouette),
        "text_explained_variance": float(text_svd.explained_variance_ratio_.sum()),
    }
    return product_documents, cluster_terms, diagnostics, tfidf_matrix


def build_shared_reviewer_graph(
    selected_reviews: pd.DataFrame,
    selected_asins: list[str],
    min_shared_reviewers: int,
) -> tuple[nx.Graph, int]:
    """Connect products that share at least a threshold number of reviewers."""

    user_product_pairs = selected_reviews[["reviewer_id", "asin"]].drop_duplicates()
    edge_weights: Counter[tuple[str, str]] = Counter()

    for _, asin_values in user_product_pairs.groupby("reviewer_id")["asin"]:
        products = sorted(set(asin_values))
        if len(products) < 2:
            continue
        for source, target in combinations(products, 2):
            edge_weights[(source, target)] += 1

    threshold = min_shared_reviewers
    filtered_edges = {
        pair: weight for pair, weight in edge_weights.items() if weight >= threshold
    }
    if not filtered_edges and edge_weights:
        threshold = 1
        filtered_edges = dict(edge_weights)

    graph = nx.Graph()
    graph.add_nodes_from(selected_asins)
    for (source, target), weight in filtered_edges.items():
        graph.add_edge(source, target, weight=weight)
    return graph, threshold


def build_graph_clusters(
    graph: nx.Graph,
    product_documents: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create graph-based embeddings, centrality features, and graph clusters."""

    ordered_asins = product_documents["asin"].tolist()
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=ordered_asins, weight="weight", dtype=float)

    if adjacency.nnz == 0:
        embedding = np.zeros((len(ordered_asins), 2))
        product_documents["graph_cluster"] = 0
        product_documents["graph_x"] = 0.0
        product_documents["graph_y"] = 0.0
        diagnostics = {
            "best_graph_k": float("nan"),
            "graph_silhouette": float("nan"),
            "graph_edges": int(graph.number_of_edges()),
            "graph_density": float(nx.density(graph)),
        }
    else:
        normalized_adjacency = normalize(adjacency, norm="l2", axis=1)
        n_components = min(16, normalized_adjacency.shape[0] - 1, normalized_adjacency.shape[1] - 1)
        n_components = max(n_components, 2)
        graph_svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        embedding = graph_svd.fit_transform(normalized_adjacency)
        best_k, best_silhouette = select_best_k(embedding, config.cluster_candidates)
        cluster_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
        product_documents["graph_cluster"] = cluster_model.fit_predict(embedding)
        product_documents["graph_x"] = embedding[:, 0]
        product_documents["graph_y"] = embedding[:, 1]
        diagnostics = {
            "best_graph_k": int(best_k),
            "graph_silhouette": float(best_silhouette),
            "graph_edges": int(graph.number_of_edges()),
            "graph_density": float(nx.density(graph)),
        }

    weighted_degree = dict(graph.degree(weight="weight"))
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {node: 0.0 for node in graph}
    clustering = (
        nx.clustering(graph, weight="weight") if graph.number_of_edges() else {node: 0.0 for node in graph}
    )

    community_sets = (
        list(nx.community.greedy_modularity_communities(graph, weight="weight"))
        if graph.number_of_edges()
        else []
    )
    community_lookup: dict[str, int] = {}
    for community_id, community_nodes in enumerate(community_sets):
        for node in community_nodes:
            community_lookup[node] = community_id

    product_documents["weighted_degree"] = product_documents["asin"].map(weighted_degree).fillna(0.0)
    product_documents["pagerank"] = product_documents["asin"].map(pagerank).fillna(0.0)
    product_documents["clustering_coefficient"] = (
        product_documents["asin"].map(clustering).fillna(0.0)
    )
    product_documents["graph_community"] = product_documents["asin"].map(community_lookup).fillna(-1).astype(int)
    diagnostics["community_count"] = int(len(community_sets))
    return product_documents, diagnostics


def run_topic_model(
    product_documents: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit a lightweight LDA model to describe the product text clusters."""

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=config.max_topic_features,
        min_df=3,
    )
    count_matrix = vectorizer.fit_transform(product_documents["document"])
    topic_model = LatentDirichletAllocation(
        n_components=config.num_topics,
        random_state=RANDOM_STATE,
        learning_method="batch",
    )
    topic_weights = topic_model.fit_transform(count_matrix)
    product_documents["dominant_topic"] = np.argmax(topic_weights, axis=1)

    feature_names = vectorizer.get_feature_names_out()
    topic_summaries: list[dict[str, Any]] = []
    for topic_id, component in enumerate(topic_model.components_):
        top_indices = component.argsort()[::-1][: config.top_terms_per_group]
        topic_summaries.append(
            {
                "topic": topic_id,
                "top_terms": ", ".join(feature_names[index] for index in top_indices),
            }
        )
    return product_documents, topic_summaries


def build_summary(
    reviews: pd.DataFrame,
    product_documents: pd.DataFrame,
    graph: nx.Graph,
    config: AnalysisConfig,
    text_diagnostics: dict[str, float],
    graph_diagnostics: dict[str, float],
    shared_reviewer_threshold: int,
) -> dict[str, Any]:
    """Assemble the project headline metrics for the notebook and CLI output."""

    rating_pagerank_corr = product_documents["mean_rating"].corr(product_documents["pagerank"])
    summary = {
        "research_question": (
            "Do products that look similar in review language also occupy similar positions "
            "in the shared-reviewer graph?"
        ),
        "total_reviews_in_category": int(len(reviews)),
        "unique_reviewers_in_category": int(reviews["reviewer_id"].nunique()),
        "unique_products_in_category": int(reviews["asin"].nunique()),
        "products_analyzed": int(len(product_documents)),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "effective_shared_reviewer_threshold": int(shared_reviewer_threshold),
        "average_rating": float(reviews["overall"].mean()),
        "cluster_alignment_nmi": float(
            normalized_mutual_info_score(
                product_documents["text_cluster"],
                product_documents["graph_cluster"],
            )
        ),
        "rating_pagerank_correlation": float(rating_pagerank_corr) if pd.notna(rating_pagerank_corr) else float("nan"),
        "config": asdict(config),
    }
    summary.update(text_diagnostics)
    summary.update(graph_diagnostics)
    return summary


def save_figures(
    reviews: pd.DataFrame,
    product_documents: pd.DataFrame,
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    """Render the figures used by the notebook and README."""

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(x=reviews["overall"].astype(int), color="#2f5d62")
    plt.title("Video Games Rating Distribution")
    plt.xlabel("Star Rating")
    plt.ylabel("Review Count")
    plt.tight_layout()
    plt.savefig(figure_dir / "rating_distribution.png", dpi=160)
    plt.close()

    top_products = product_documents.head(config.top_products_in_plot).sort_values("review_count")
    plt.figure(figsize=(9, 6))
    plt.barh(top_products["asin"], top_products["review_count"], color="#5e8c61")
    plt.title("Most Reviewed Products in the Analysis Subset")
    plt.xlabel("Review Count")
    plt.ylabel("ASIN")
    plt.tight_layout()
    plt.savefig(figure_dir / "top_products.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        product_documents["text_x"],
        product_documents["text_y"],
        c=product_documents["text_cluster"],
        cmap="tab10",
        s=50,
        alpha=0.85,
    )
    plt.title("Text-Derived Product Clusters")
    plt.xlabel("TF-IDF SVD 1")
    plt.ylabel("TF-IDF SVD 2")
    plt.colorbar(scatter, label="Text Cluster")
    plt.tight_layout()
    plt.savefig(figure_dir / "text_clusters.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        product_documents["pagerank"],
        product_documents["mean_rating"],
        s=product_documents["review_count"] * 2,
        alpha=0.7,
        color="#8f2d56",
    )
    plt.title("Graph Importance versus Average Rating")
    plt.xlabel("PageRank")
    plt.ylabel("Average Rating")
    plt.tight_layout()
    plt.savefig(figure_dir / "pagerank_vs_rating.png", dpi=160)
    plt.close()


def save_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    product_documents: pd.DataFrame,
    cluster_terms: list[dict[str, Any]],
    topic_terms: list[dict[str, Any]],
) -> None:
    """Persist the main analysis tables for reuse in the notebook or README."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    product_documents.to_csv(output_dir / "product_results.csv", index=False)
    pd.DataFrame(cluster_terms).to_csv(output_dir / "text_cluster_terms.csv", index=False)
    pd.DataFrame(topic_terms).to_csv(output_dir / "topic_terms.csv", index=False)


def run_analysis(
    review_path: str | Path,
    output_dir: str | Path,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Run the full semantic-versus-structural comparison workflow."""

    review_path = Path(review_path)
    output_dir = Path(output_dir)
    config = config or AnalysisConfig()

    reviews = load_reviews(review_path)
    selected_reviews, product_documents = prepare_product_documents(reviews, config)
    product_documents, cluster_terms, text_diagnostics, _ = run_text_analysis(product_documents, config)

    graph, shared_reviewer_threshold = build_shared_reviewer_graph(
        selected_reviews,
        product_documents["asin"].tolist(),
        config.min_shared_reviewers,
    )
    product_documents, graph_diagnostics = build_graph_clusters(graph, product_documents, config)
    product_documents, topic_terms = run_topic_model(product_documents, config)

    summary = build_summary(
        reviews=reviews,
        product_documents=product_documents,
        graph=graph,
        config=config,
        text_diagnostics=text_diagnostics,
        graph_diagnostics=graph_diagnostics,
        shared_reviewer_threshold=shared_reviewer_threshold,
    )

    save_outputs(output_dir, summary, product_documents, cluster_terms, topic_terms)
    save_figures(reviews, product_documents, output_dir, config)

    community_sizes = (
        product_documents.loc[product_documents["graph_community"] >= 0, "graph_community"]
        .value_counts()
        .rename_axis("community")
        .reset_index(name="size")
        .sort_values("size", ascending=False)
    )

    return {
        "summary": summary,
        "product_documents": product_documents,
        "cluster_terms": pd.DataFrame(cluster_terms),
        "topic_terms": pd.DataFrame(topic_terms),
        "community_sizes": community_sizes,
    }
