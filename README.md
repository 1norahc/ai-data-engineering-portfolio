# AI & Data Engineering Portfolio

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![MIT](https://img.shields.io/badge/License-MIT-green)

A collection of three end-to-end machine learning projects completed as part of the MIT Applied Data Science Program, each extended into a production-ready portfolio piece with clean code, interactive dashboards, and full documentation.

---

## Projects

### [Project 1 — Customer Personality Segmentation](./Project1%20-%20Customer%20Personality%20Segmentation/)

> **Domain:** Retail & Marketing | **Technique:** Unsupervised Learning — K-Means Clustering

Segments 2,240 retail customers into 5 distinct personality groups based on demographics, spending behavior, and campaign response history. Enables personalized marketing strategies and resource allocation.

| Cluster | Segment Name | Size | Avg Income | Key Trait |
|---------|-------------|------|------------|-----------|
| 0 | Affluent Traditionalists | 452 | $73,964 | Store/catalog shoppers |
| 1 | Low-Income Minimal Spenders | 998 | $34,692 | Budget-conscious, web users |
| 2 | Premium Heavy Spenders | 169 | $81,738 | Highest wine & meat spend |
| 3 | Dormant Families | 21 | $45,242 | Recently inactive, children |
| 4 | Digital-Savvy Spenders | 600 | $56,992 | Web-first, teenagers |

**Highlights:** K-Means with elbow + silhouette analysis, feature engineering, interactive Streamlit dashboard.

---

### [Project 2 — Lead Conversion Prediction](./Project2%20-%20Potential%20Customers%20Prediction/)

> **Domain:** EdTech / Sales | **Technique:** Supervised Learning — Decision Tree & Random Forest

Predicts which of 4,612 leads from EdTech startup ExtraaLearn will convert to paying customers. Identifies top conversion drivers to optimize sales resource allocation.

| Model | Accuracy | ROC-AUC | PR-AUC |
|-------|----------|---------|--------|
| Decision Tree (pruned) | 86.78% | 0.922 | — |
| Random Forest (tuned) | 85.05% | 0.930 | 0.852 |

**Top features:** Time spent on website, first interaction channel, profile completion, page views.

**Highlights:** Cost-complexity pruning, RandomizedSearchCV, lead scoring Streamlit app.

---

### [Project 3 — Amazon Product Recommendation System](./Project3%20-%20Amazon%20Product%20Recommendation%20System/)

> **Domain:** E-commerce | **Technique:** Collaborative Filtering — KNN & SVD Matrix Factorization

Builds and compares four recommendation algorithms on 7.8M Amazon Electronics ratings, filtered to 65,290 interactions from active users. SVD Matrix Factorization achieves the best performance.

| Model | RMSE | Precision@10 | Recall@10 | F1@10 |
|-------|------|--------------|-----------|-------|
| Rank-Based (Baseline) | — | — | — | — |
| User-User CF | 0.9741 | 0.836 | 0.895 | 0.864 |
| Item-Item CF | 0.9752 | 0.830 | 0.893 | 0.860 |
| **SVD (Best)** | **0.9039** | **0.838** | **0.877** | **0.857** |

**Highlights:** Surprise library, GridSearchCV tuning, sparsity handling, real-time recommendation Streamlit app.

---

## Repository Structure

```
ai-data-engineering-portfolio/
├── README.md
├── .gitignore
│
├── Project1 - Customer Personality Segmentation/
│   ├── README.md
│   ├── requirements.txt
│   ├── main.py                 # Full pipeline runner
│   ├── app.py                  # Streamlit dashboard
│   ├── src/
│   │   ├── preprocessing.py
│   │   ├── eda.py
│   │   ├── clustering.py
│   │   └── visualization.py
│   └── reports/figures/
│
├── Project2 - Potential Customers Prediction/
│   ├── README.md
│   ├── requirements.txt
│   ├── main.py
│   ├── app.py                  # Lead Scoring tool
│   ├── data/ExtraaLearn.csv
│   ├── src/
│   │   ├── preprocessing.py
│   │   ├── eda.py
│   │   ├── models.py
│   │   └── evaluation.py
│   └── reports/figures/
│
└── Project3 - Amazon Product Recommendation System/
    ├── README.md
    ├── requirements.txt
    ├── main.py
    ├── app.py                  # Recommendation engine UI
    ├── data/ratings_Electronics.csv
    ├── src/
    │   ├── preprocessing.py
    │   ├── eda.py
    │   ├── recommenders.py
    │   └── evaluation.py
    └── reports/figures/
```

---

## Quick Start

Each project is self-contained. Navigate into any project folder and follow its `README.md`.

```bash
# Example: run Project 1
cd "Project1 - Customer Personality Segmentation"
pip install -r requirements.txt
python main.py              # runs full analysis pipeline
streamlit run app.py        # launches interactive dashboard
```

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Processing | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `scikit-surprise` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |
| Model Persistence | `joblib` |

---

## About

These projects were completed as part of the **MIT Applied Data Science Program**. Each achieved a perfect score and has been extended here into production-ready code with modular architecture, interactive frontends, and comprehensive documentation.

> **Rajan** — Data Scientist / AI Engineer