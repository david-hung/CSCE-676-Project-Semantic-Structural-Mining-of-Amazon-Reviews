"""Generate the curated final notebook for the Amazon Video Games project."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amazon_video_games_project.analysis import AnalysisConfig, run_analysis


def _code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def _markdown_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def main() -> None:
    data_path = PROJECT_ROOT / "data/raw/Video_Games_5.json.gz"
    output_dir = PROJECT_ROOT / "outputs/video_games"
    notebook_path = PROJECT_ROOT / "main_notebook.ipynb"

    if not data_path.exists():
        raise FileNotFoundError(
            "Expected the dataset at data/raw/Video_Games_5.json.gz before building the final notebook."
        )

    config = AnalysisConfig(
        top_products=400,
        min_reviews_per_product=15,
        min_shared_reviewers=2,
        num_topics=6,
    )
    results = run_analysis(data_path, output_dir, config)
    summary = results["summary"]
    cluster_terms = results["cluster_terms"]
    topic_terms = results["topic_terms"]

    text_cluster_lines = "\n".join(
        f"- Text cluster {int(row.cluster)} is characterized by `{row.top_terms}`."
        for row in cluster_terms.itertuples(index=False)
    )
    topic_lines = "\n".join(
        f"- Topic {int(row.topic)} is summarized by `{row.top_terms}`."
        for row in topic_terms.itertuples(index=False)
    )

    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
        }
    )

    notebook.cells = [
        _markdown_cell(
            f"""
            # Semantic vs Structural Signals in Amazon Video Games Reviews

            ## Motivation

            Amazon reviews expose two different kinds of product similarity. The words inside reviews tell us how customers describe a product, while the overlap of reviewers tells us which products are consumed by the same audience. Those signals need not agree, and that mismatch is interesting in its own right because it reveals whether "talked about similarly" and "used by similar people" are actually the same thing.

            This final notebook focuses on one clean question inside one dataset: the Amazon Review Data 2018 `Video_Games_5` category. The goal is not to show every experiment from the project, but to give one coherent answer supported by text mining and graph mining.

            ## Research Question

            **Do products that look similar in review language also occupy similar positions in a shared-reviewer product graph?**

            ## Short Answer

            For this dataset, the answer is mostly **no**. On the final subset of **{summary['products_analyzed']}** frequently reviewed products, the semantic view prefers **{summary['best_text_k']}** broad text clusters, the structural view prefers **{summary['best_graph_k']}** graph clusters, and the alignment between those partitions is only **{summary['cluster_alignment_nmi']:.3f}** by normalized mutual information. Review language and reviewer-overlap structure are therefore related, but they are far from interchangeable.
            """
        ),
        _markdown_cell(
            f"""
            ## Dataset and Method

            The raw file contains **{summary['total_reviews_in_category']:,}** reviews, **{summary['unique_reviewers_in_category']:,}** reviewers, and **{summary['unique_products_in_category']:,}** products in the Video Games category. To keep the comparison interpretable, I restrict the final analysis to the **{summary['products_analyzed']}** most-reviewed products that have at least 15 reviews. This creates a dense enough text corpus and reviewer graph to compare the two views directly.

            The pipeline is intentionally compact:

            - Aggregate all review text for each product into one product-level document.
            - Build TF-IDF vectors and cluster those documents with KMeans.
            - Build a shared-reviewer graph where an edge means at least **{summary['effective_shared_reviewer_threshold']}** shared reviewers.
            - Extract graph-based structure with PageRank, communities, and graph clustering.
            - Use LDA topic modeling as an interpretation layer rather than as the main result.

            This uses the core course ideas of text mining, clustering, and graph analysis, plus topic modeling as the extension method.
            """
        ),
        _code_cell(
            """
            from pathlib import Path
            import sys

            import pandas as pd
            from IPython.display import Image, display

            PROJECT_ROOT = Path.cwd().resolve()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent

            SRC_DIR = PROJECT_ROOT / "src"
            if str(SRC_DIR) not in sys.path:
                sys.path.insert(0, str(SRC_DIR))

            DATA_PATH = PROJECT_ROOT / "data/raw/Video_Games_5.json.gz"
            OUTPUT_DIR = PROJECT_ROOT / "outputs/video_games"

            if not DATA_PATH.exists():
                raise FileNotFoundError(
                    "Dataset not found. Place it at data/raw/Video_Games_5.json.gz "
                    "or run python scripts/download_video_games_data.py."
                )

            from amazon_video_games_project.analysis import AnalysisConfig, run_analysis

            pd.set_option("display.max_colwidth", 120)
            """
        ),
        _code_cell(
            """
            config = AnalysisConfig(
                top_products=400,
                min_reviews_per_product=15,
                min_shared_reviewers=2,
                num_topics=6,
            )

            results = run_analysis(DATA_PATH, OUTPUT_DIR, config)
            summary = results["summary"]
            product_documents = results["product_documents"]
            cluster_terms = results["cluster_terms"]
            topic_terms = results["topic_terms"]
            community_sizes = results["community_sizes"]
            """
        ),
        _markdown_cell(
            f"""
            ## Headline Findings

            Four quantitative results drive the answer:

            - The shared-reviewer graph is dense: **{summary['graph_edges']:,}** edges across **{summary['graph_nodes']}** products for a graph density of **{summary['graph_density']:.3f}**.
            - The semantic pipeline finds **{summary['best_text_k']}** text clusters, but the graph view breaks the same products into **{summary['best_graph_k']}** structural clusters and **{summary['community_count']}** communities.
            - The alignment between the text clusters and graph clusters is only **{summary['cluster_alignment_nmi']:.3f}**, which is low.
            - Product centrality and average rating are only weakly related: the PageRank-to-rating correlation is **{summary['rating_pagerank_correlation']:.3f}**.

            Together, those numbers suggest that the language customers use and the audience pathways that connect products capture different dimensions of the category.
            """
        ),
        _code_cell(
            """
            summary_frame = pd.Series(
                {
                    "reviews in full category": f"{summary['total_reviews_in_category']:,}",
                    "reviewers in full category": f"{summary['unique_reviewers_in_category']:,}",
                    "products in full category": f"{summary['unique_products_in_category']:,}",
                    "products analyzed": summary["products_analyzed"],
                    "shared-reviewer graph edges": f"{summary['graph_edges']:,}",
                    "graph density": round(summary["graph_density"], 3),
                    "best text cluster count": summary["best_text_k"],
                    "best graph cluster count": summary["best_graph_k"],
                    "cluster alignment NMI": round(summary["cluster_alignment_nmi"], 3),
                    "rating/PageRank correlation": round(summary["rating_pagerank_correlation"], 3),
                },
                name="value",
            ).to_frame()
            summary_frame
            """
        ),
        _markdown_cell(
            """
            ## Exploratory Data Analysis

            Before comparing semantics and structure, it helps to understand the category itself. The rating distribution shows that Video Games reviews are strongly skewed toward positive scores, which is common for self-selected review platforms. The second figure shows how concentrated the analysis subset is: a relatively small set of products attracts a large share of all activity.
            """
        ),
        _code_cell(
            """
            subset_overview = pd.DataFrame(
                [
                    {
                        "subset_products": len(product_documents),
                        "avg_reviews_per_product": round(product_documents["review_count"].mean(), 1),
                        "median_reviews_per_product": round(product_documents["review_count"].median(), 1),
                        "avg_rating_in_subset": round(product_documents["mean_rating"].mean(), 3),
                    }
                ]
            )

            display(Image(filename=str(OUTPUT_DIR / "figures/rating_distribution.png")))
            display(Image(filename=str(OUTPUT_DIR / "figures/top_products.png")))
            subset_overview
            """
        ),
        _markdown_cell(
            f"""
            ## Semantic View: What the Review Text Says

            The semantic pipeline clusters product-level review documents using TF-IDF. The best solution on this subset uses **{summary['best_text_k']}** clusters with a silhouette score of **{summary['text_silhouette']:.3f}**. The resulting groups are interpretable rather than arbitrary:

            {text_cluster_lines}

            The main pattern is that one large cluster captures general game-discussion language, while smaller clusters isolate hardware-heavy review vocabulary such as controllers, Wii accessories, and gaming mice. That tells us the text is sensitive to product type and usage context.
            """
        ),
        _code_cell(
            """
            semantic_profile = (
                product_documents.groupby("text_cluster")
                .agg(
                    products=("asin", "size"),
                    avg_reviews=("review_count", "mean"),
                    avg_rating=("mean_rating", "mean"),
                    avg_pagerank=("pagerank", "mean"),
                )
                .round(3)
                .sort_values("products", ascending=False)
            )

            display(Image(filename=str(OUTPUT_DIR / "figures/text_clusters.png")))
            display(cluster_terms)
            semantic_profile
            """
        ),
        _markdown_cell(
            f"""
            ## Structural View: Who Reviews the Same Products

            The structural view throws away the words and keeps only reviewer overlap. Two products are connected when they share at least **{summary['effective_shared_reviewer_threshold']}** reviewers. On this subset, that produces a surprisingly dense network with **{summary['graph_edges']:,}** weighted edges and **{summary['community_count']}** graph communities.

            This matters because reviewer overlap is not the same as textual similarity. A reviewer can move across products for many reasons: franchise loyalty, hardware compatibility, gift shopping, or platform preference. The graph therefore captures audience pathways that are partly hidden from the text model.
            """
        ),
        _code_cell(
            """
            graph_overview = pd.DataFrame(
                [
                    {
                        "nodes": summary["graph_nodes"],
                        "edges": summary["graph_edges"],
                        "density": round(summary["graph_density"], 3),
                        "graph communities": summary["community_count"],
                        "best graph clusters": summary["best_graph_k"],
                    }
                ]
            )

            top_pagerank = product_documents[
                ["asin", "review_count", "mean_rating", "pagerank", "weighted_degree", "graph_community"]
            ].sort_values("pagerank", ascending=False).head(15)

            display(graph_overview)
            display(top_pagerank)
            display(community_sizes.head(10))
            display(Image(filename=str(OUTPUT_DIR / "figures/pagerank_vs_rating.png")))
            """
        ),
        _markdown_cell(
            f"""
            ## Direct Comparison: Do the Two Views Agree?

            This is the core question of the project. If the semantic and structural views told the same story, products grouped together by review language would also be grouped together by reviewer overlap. That is not what happens here.

            The alignment score between the text clusters and graph clusters is only **{summary['cluster_alignment_nmi']:.3f}**. That is far closer to weak agreement than to a one-to-one match. In other words, products that *sound* similar in reviews do not reliably occupy the same position in the shared-reviewer network.

            This is the central conclusion of the notebook: semantic similarity is useful, but it cannot substitute for structural information about who reviews what.
            """
        ),
        _code_cell(
            """
            alignment_table = pd.crosstab(
                product_documents["text_cluster"],
                product_documents["graph_cluster"],
                rownames=["text cluster"],
                colnames=["graph cluster"],
            )

            alignment_share = alignment_table.div(alignment_table.sum(axis=1), axis=0).round(3)
            dominant_graph_share = alignment_share.max(axis=1).rename("largest graph-cluster share")

            display(alignment_table)
            pd.concat([alignment_share, dominant_graph_share], axis=1)
            """
        ),
        _markdown_cell(
            f"""
            ## Interpretation Extension: Topic Modeling

            Topic modeling is not the final answer, but it helps explain why the semantic clusters look the way they do. The LDA model recovers recurring themes that line up with recognizable subdomains of the category, including gameplay language, Nintendo/Wii-oriented discussion, shooter-specific vocabulary, and hardware/controller talk.

            {topic_lines}

            This extension is useful because it turns the text clusters from a geometric result into a substantive interpretation.
            """
        ),
        _code_cell(
            """
            topic_by_cluster = pd.crosstab(
                product_documents["text_cluster"],
                product_documents["dominant_topic"],
                normalize="index",
            ).round(3)

            display(topic_terms)
            topic_by_cluster
            """
        ),
        _markdown_cell(
            """
            ## Limitations and Ethical Notes

            This analysis is informative, but it is not neutral or complete.

            - The Amazon 5-core dataset keeps only products and users with at least five interactions, so it already filters the market before analysis begins.
            - The notebook works at the product level without metadata such as product title or platform, which limits human interpretability of ASIN-level tables.
            - Amazon reviews are voluntary and often polarized, so the text reflects who chooses to post rather than all customers.
            - Reviewer overlap may reflect popularity, compatibility, or franchise spillover rather than genuine product similarity.

            These caveats do not invalidate the result, but they do explain why semantic and structural signals should be interpreted as complementary rather than definitive.
            """
        ),
        _markdown_cell(
            f"""
            ## Conclusion

            The final answer to the project question is clear: **products that look similar in review language do not consistently occupy the same place in the shared-reviewer graph**. The low alignment score of **{summary['cluster_alignment_nmi']:.3f}** shows that the two views disagree more often than they coincide.

            The text model mainly separates broad product types and recurring discussion themes. The graph model instead captures audience overlap and product pathways. Those are different mechanisms, and that difference is exactly why combining semantic and structural mining is more informative than using either one alone.

            A second practical takeaway is that graph importance is only weakly connected to perceived product quality in this subset: the rating-to-PageRank correlation is just **{summary['rating_pagerank_correlation']:.3f}**. Popularly connected products are not simply the highest-rated ones.
            """
        ),
    ]

    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(nbf.writes(notebook), encoding="utf-8")
    print(f"Wrote curated notebook to {notebook_path}")


if __name__ == "__main__":
    main()
