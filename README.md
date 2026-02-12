# CSCE 676 Project: Semantic & Structural Mining of Amazon Reviews

## Overview

This project analyzes the Amazon Reviews dataset using text mining and graph mining techniques. It combines course methods from CSCE 676—such as embeddings, clustering, and graph analysis—with advanced approaches including transformer-based embeddings and topic modeling.

The goal is to explore how semantic information from review text and structural information from user–product networks can be integrated to uncover meaningful product insights.

---

## Objectives

- Perform exploratory data analysis on large-scale review data
- Apply text mining techniques (TF-IDF, embeddings)
- Construct and analyze user–product or product–product graphs
- Compare traditional methods with advanced embedding approaches
- Investigate relationships between semantic similarity and graph structure

---

## Dataset

**Dataset:** Amazon Reviews Dataset (Stanford SNAP)

This dataset includes:
- User IDs
- Product IDs
- Ratings (1–5)
- Review text
- Timestamps
- Optional product graph data (co-purchase / co-view links)

⚠️ The raw dataset is **not included** in this repository due to size and licensing considerations.  
Please download it directly from the official SNAP website.

---

## Methods

### Course Techniques
- Text vectorization (TF-IDF / embeddings)
- Clustering
- Graph construction and degree analysis
- Community detection

### Beyond-Course Techniques
- Transformer-based sentence embeddings
- Topic modeling (e.g., LDA or BERTopic)
- Advanced graph representation learning (optional extension)

---

## Ethical & Bias Considerations

- Reviews are self-selected and may reflect sentiment bias.
- Popular products may dominate analysis.
- No personally identifiable information is used.
- Results are analyzed at aggregate levels only.

---

## License

This repository is licensed under the MIT License.  
The Amazon dataset remains subject to its original licensing terms.

---

## Author

CSCE 676 – Data Mining and Analysis  
Texas A&M University
