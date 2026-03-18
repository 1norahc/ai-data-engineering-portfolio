# Lead Conversion Prediction — ExtraaLearn

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![MIT](https://img.shields.io/badge/MIT%20Program-100%25-green)

> Predict which leads will convert to paying customers for an EdTech startup, enabling smarter sales resource allocation.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Project Structure](#project-structure)
6. [Installation & Usage](#installation--usage)
7. [Key Findings](#key-findings)
8. [Technologies](#technologies)

---

## Problem Statement

ExtraaLearn, an EdTech startup offering courses in emerging technologies, generates many leads through various digital channels. The challenge: only ~30% of leads convert to paying customers. Manually identifying high-potential leads is time-consuming and subjective.

**Goal:** Build a classification model to:
1. Predict which leads are likely to convert
2. Identify the key factors that drive conversion
3. Enable sales reps to prioritise high-probability leads

---

## Dataset

**File:** `data/ExtraaLearn.csv`
**Size:** 4,612 leads × 15 columns

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Lead age (18–63) |
| `current_occupation` | Categorical | Professional / Unemployed / Student |
| `first_interaction` | Categorical | Website / Mobile App |
| `profile_completed` | Ordinal | Low / Medium / High |
| `website_visits` | Numeric | Number of website visits |
| `time_spent_on_website` | Numeric | Total seconds spent on site |
| `page_views_per_visit` | Numeric | Avg pages viewed per session |
| `last_activity` | Categorical | Email / Phone / Website Activity |
| `print_media_type1/2` | Binary | Exposed to print media |
| `digital_media` | Binary | Exposed to digital ads |
| `educational_channels` | Binary | Reached via educational channels |
| `referral` | Binary | Referred by existing customer |
| `status` | **Target** | 1 = Converted, 0 = Not Converted |

**Class balance:** 70% not converted (3,235), 30% converted (1,377)

---

## Methodology

```
Raw Data (4,612 leads)
   │
   ▼
Outlier Treatment       ← 1st–99th percentile clipping on numeric features
   │
   ▼
Feature Encoding        ← Ordinal encode profile_completed (Low=0, Med=1, High=2)
   │                       Create is_professional binary flag
   ▼
Train/Test Split        ← 80/20 stratified split
   │
   ▼
ColumnTransformer       ← StandardScaler (numeric) + OneHotEncoder (categorical)
   │
   ├── Decision Tree     ← Cost-complexity pruning (best ccp_alpha via 5-fold CV)
   │
   └── Random Forest     ← 400 trees + RandomizedSearchCV (25 iterations, 5-fold CV)
                            Search: max_depth, min_samples_split, min_samples_leaf, max_features
```

---

## Results

### Model Comparison

| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|----------|---------|--------|-----------|--------|-----|
| Decision Tree (pruned) | **86.78%** | 0.922 | — | 0.82 | 0.73 | 0.77 |
| Random Forest (tuned) | 85.05% | **0.930** | **0.852** | 0.80 | 0.74 | 0.77 |

Both models perform well. Decision Tree is more interpretable; Random Forest generalises better by ROC-AUC.

### Top 5 Predictive Features

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `time_spent_on_website` | 27% |
| 2 | `first_interaction_Website` | 26% |
| 3 | `profile_completed` | 23% |
| 4 | `page_views_per_visit` | 11% |
| 5 | `last_activity_Phone Activity` | 7% |

---

## Project Structure

```
Project2 - Potential Customers Prediction/
├── README.md
├── requirements.txt
├── main.py                    # Full pipeline runner (CLI)
├── app.py                     # Streamlit dashboard + lead scorer
├── data/
│   └── ExtraaLearn.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # load_data, clip_outliers, encode_ordinal, build_preprocessor
│   ├── eda.py                 # EDA plot functions
│   ├── models.py              # Decision Tree + Random Forest training
│   └── evaluation.py         # Metrics, ROC curves, feature importance
└── reports/
    ├── model_results.csv      # Generated comparison table
    ├── models/                # Saved .joblib model files
    └── figures/               # Generated PNG plots
```

---

## Installation & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full training pipeline
python main.py
# → trains both models, prints comparison table, saves plots and models

# 3. Launch the interactive Streamlit dashboard
streamlit run app.py
# → includes live lead scoring tool (enter lead details → get conversion probability)
```

---

## Key Findings

1. **Website engagement is the strongest signal** — leads who spend more time on the site convert at much higher rates
2. **Profile completion matters** — leads with complete profiles are significantly more likely to convert
3. **Phone activity outperforms email** — calling leads is more effective than email nurturing
4. **Professional occupation** correlates with higher conversion vs. students/unemployed
5. **Referrals and digital media** show the highest channel-level conversion rates

---

## Technologies

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, DecisionTree, RandomForest, GridSearch |
| `matplotlib` / `seaborn` | Static visualisations |
| `plotly` | Interactive Streamlit charts |
| `streamlit` | Web dashboard + lead scorer |
| `joblib` | Model serialisation |
