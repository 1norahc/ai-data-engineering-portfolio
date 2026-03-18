# Customer Personality Segmentation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![MIT](https://img.shields.io/badge/MIT%20Program-100%25-green)

> Segment retail customers into actionable personality groups using K-Means clustering, enabling targeted marketing and personalised campaigns.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Project Structure](#project-structure)
6. [Installation & Usage](#installation--usage)
7. [Business Insights](#business-insights)
8. [Technologies](#technologies)

---

## Problem Statement

A retail company wants to understand its customers better to design effective marketing campaigns. Rather than treating all customers equally, the goal is to identify distinct customer **personality segments** based on:

- Demographic data (age, income, education, household)
- 2-year purchasing history across 6 product categories
- Response to 5 previous marketing campaigns
- Preferred shopping channels (web, catalog, store)

---

## Dataset

**File:** `Customer_Personality_Segmentation.csv`
**Size:** 2,240 customers × 29 features (tab-separated)

| Feature Group | Columns |
|---------------|---------|
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Spending | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Campaigns | `AcceptedCmp1`–`AcceptedCmp5`, `Response` |
| Channels | `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumDealsPurchases` |
| Engagement | `Recency`, `NumWebVisitsMonth`, `Complain` |

**Engineered features:** `Age`, `Tenure` (days enrolled), `TotalSpending`, `TotalPurchases`, `TotalCampaigns`, `SpendingPerPurchase`, `HasChildren`

---

## Methodology

```
Raw Data
   │
   ▼
Data Cleaning          ← Fill 24 missing Income values (median imputation)
   │                      Remove extreme outliers (Income > $200k, Year_Birth < 1900)
   ▼                      Drop zero-variance columns (Z_CostContact, Z_Revenue)
Feature Engineering    ← Age, Tenure, TotalSpending, TotalPurchases, HasChildren
   │
   ▼
StandardScaler         ← Z-score normalisation before distance-based clustering
   │
   ▼
Optimal-k Selection    ← Elbow Method (inertia) + Silhouette Score for k = 2..10
   │
   ▼
K-Means (k=5)          ← 10 random initialisations, random_state=42
   │
   ▼
Cluster Profiling      ← Mean metrics per cluster, business name assignment
```

---

## Results

### Cluster Profiles

| Cluster | Name | Size | Avg Income | Avg Spending | Key Trait |
|---------|------|------|-----------|-------------|-----------|
| 0 | Affluent Traditionalists | 452 | $73,964 | High | Store & catalog preference |
| 1 | Low-Income Minimal Spenders | 998 | $34,692 | Low | Budget-conscious, web users |
| 2 | Premium Heavy Spenders | 169 | $81,738 | Highest (wine $877) | Multi-channel, no kids |
| 3 | Dormant Families | 21 | $45,242 | Low | Recently inactive, children |
| 4 | Digital-Savvy Spenders | 600 | $56,992 | High | Web-dominant, teenagers |

### Model Metrics

| Metric | Value |
|--------|-------|
| Algorithm | K-Means (k=5) |
| Inertia (WCSS) | 34,105 |
| Silhouette Score | 0.187 |
| Fit Time | < 0.1 s |

---

## Project Structure

```
Project1 - Customer Personality Segmentation/
├── README.md
├── requirements.txt
├── main.py                    # Full pipeline runner (CLI)
├── app.py                     # Streamlit interactive dashboard
├── Customer_Personality_Segmentation.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # load_data, clean_data, engineer_features, get_feature_matrix
│   ├── eda.py                 # EDA plot functions
│   ├── clustering.py          # compute_elbow_metrics, fit_kmeans, assign_clusters
│   └── visualization.py      # Post-clustering visualisations
└── reports/
    ├── customer_segments.csv  # Generated output
    └── figures/               # Generated PNG plots
```

---

## Installation & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full analysis pipeline
python main.py
# → prints cluster summary, saves reports/customer_segments.csv and all plots

# 3. Launch the interactive Streamlit dashboard
streamlit run app.py
```

---

## Business Insights

| Segment | Priority | Recommended Action |
|---------|----------|-------------------|
| Premium Heavy Spenders | HIGH | VIP loyalty program, exclusive wine/meat bundles |
| Affluent Traditionalists | HIGH | Premium catalogs, in-store loyalty cards |
| Digital-Savvy Spenders | MEDIUM | Social media ads, mobile UX optimisation |
| Low-Income Minimal Spenders | MEDIUM | Value packs, discount loyalty points |
| Dormant Families | LOW | Win-back campaign; evaluate re-engagement ROI |

---

## Technologies

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | StandardScaler, KMeans, silhouette_score |
| `matplotlib` / `seaborn` | Static visualisations |
| `plotly` | Interactive Streamlit charts |
| `streamlit` | Interactive web dashboard |
| `yellowbrick` | KElbow visualiser (notebook) |
