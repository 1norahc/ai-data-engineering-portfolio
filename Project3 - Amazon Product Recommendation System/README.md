# Amazon Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-surprise](https://img.shields.io/badge/Surprise-1.1%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![MIT](https://img.shields.io/badge/MIT%20Program-100%25-green)

> Build, compare, and deploy four recommendation algorithms on Amazon Electronics ratings data — from a popularity baseline to SVD matrix factorisation.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Project Structure](#project-structure)
6. [Installation & Usage](#installation--usage)
7. [How to Get Recommendations](#how-to-get-recommendations)
8. [Key Findings](#key-findings)
9. [Technologies](#technologies)

---

## Problem Statement

Amazon's Electronics category contains millions of product ratings from diverse users. The challenge: identify which products a user is most likely to enjoy, from a catalogue of thousands of items they've never interacted with.

**Goal:** Build and compare multiple recommendation approaches that:
1. Personalise product suggestions based on user behaviour
2. Handle the extreme sparsity of the user-item matrix
3. Outperform a simple popularity-based baseline

---

## Dataset

**File:** `data/ratings_Electronics.csv`
**Raw size:** 7,824,482 ratings
**Filtered size:** ~65,290 interactions (after preprocessing)

| Column | Description |
|--------|-------------|
| `userId` | Amazon customer identifier |
| `productId` | Amazon product ASIN |
| `rating` | Star rating (1–5) |
| `timestamp` | Unix timestamp of the rating |

**Filtering logic:**
- Keep users with **≥ 50 ratings** (ensures enough signal per user)
- Keep products with **≥ 5 ratings** (avoids obscure products)

**Data characteristics:**
- Mean rating: **4.29** (positive skew — users tend to rate items they like)
- Median: **5.0**
- Matrix sparsity: **> 99%**

---

## Methodology

```
Raw Data (7.8M ratings)
   │
   ▼
Filter Active Users        ← Keep users with ≥ 50 ratings
Filter Popular Products    ← Keep products with ≥ 5 ratings
   │
   ▼
Surprise Dataset           ← Reader(rating_scale=(1,5)) → Dataset
   │
   ▼
Train/Test Split (80/20)
   │
   ├── Rank-Based           No ML; sort by avg rating + min interactions
   │
   ├── User-User CF         KNNBasic (cosine, user_based=True)
   │                        Best params: k=60, min_k=5
   │
   ├── Item-Item CF         KNNBasic (cosine, user_based=False)
   │                        Best params: k=30, min_k=6
   │
   └── SVD                  Matrix Factorisation
                            Best params: n_epochs=30, lr=0.005, reg=0.02
```

---

## Results

### Model Comparison

| Model | RMSE | Precision@10 | Recall@10 | F1@10 |
|-------|------|--------------|-----------|-------|
| Rank-Based (Baseline) | — | — | — | — |
| User-User CF (tuned) | 0.9741 | 0.836 | 0.895 | 0.864 |
| Item-Item CF (tuned) | 0.9752 | 0.830 | 0.893 | 0.860 |
| **SVD (Best)** | **0.9039** | **0.838** | **0.877** | **0.857** |

SVD achieves the lowest RMSE, outperforming both neighbourhood-based approaches.

---

## Project Structure

```
Project3 - Amazon Product Recommendation System/
├── README.md
├── requirements.txt
├── main.py                    # Full pipeline CLI runner
├── app.py                     # Streamlit dashboard + recommendation UI
├── data/
│   └── ratings_Electronics.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # load_data, filter_*, prepare_surprise_data
│   ├── eda.py                 # EDA plot functions
│   ├── recommenders.py        # All 4 model implementations + get_recommendations
│   └── evaluation.py         # RMSE, precision_recall_at_k, model comparison
└── reports/
    ├── model_results.csv      # Generated comparison table
    └── figures/               # Generated PNG plots
```

---

## Installation & Usage

```bash
# 1. Install dependencies
# Note: scikit-surprise may require a C compiler (or use conda)
pip install -r requirements.txt

# If pip fails for scikit-surprise, try:
conda install -c conda-forge scikit-surprise

# 2. Run the full pipeline (loads 500k rows by default for speed)
python main.py

# Run with more data (slower but more accurate)
python main.py --nrows 0      # loads all 7.8M rows

# 3. Launch the interactive Streamlit dashboard
streamlit run app.py
```

---

## How to Get Recommendations

```python
from src.preprocessing import run_preprocessing_pipeline
from src.recommenders import train_svd, get_recommendations_for_user

# Load and prepare data
data = run_preprocessing_pipeline(nrows=200_000)
svd = train_svd(data["trainset"])

# Get top-10 recommendations for a user
recs = get_recommendations_for_user(
    algo=svd,
    user_id="A3LDPF5FMB782Z",
    df=data["df_filtered"],
    n=10,
    threshold=3.5
)
print(recs)
```

---

## Key Findings

1. **SVD beats neighbourhood methods** — latent factor decomposition captures deeper preference patterns than surface-level similarity
2. **Item-Item CF handles new users better** — item similarity is more stable and requires less user history
3. **High positive rating bias** — most users only rate items they like (mean 4.29/5), which inflates precision metrics
4. **Filtering is critical** — removing inactive users and unpopular products reduces noise and improves all models significantly
5. **Recommendation threshold 3.5+** — products predicted below 3.5 stars are unlikely to be appreciated by the user

---

## Technologies

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data manipulation and filtering |
| `scikit-surprise` | KNNBasic, SVD, GridSearchCV, train_test_split |
| `matplotlib` / `seaborn` | Static visualisations |
| `plotly` | Interactive Streamlit charts |
| `streamlit` | Web dashboard + recommendation UI |
