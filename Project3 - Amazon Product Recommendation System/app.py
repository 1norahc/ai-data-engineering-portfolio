"""
app.py — Amazon Product Recommendation System  ·  Streamlit Dashboard

Launch with:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import run_preprocessing_pipeline
from src.recommenders import (
    get_rank_based_recommendations,
    train_user_user_cf,
    train_item_item_cf,
    train_svd,
    get_recommendations_for_user,
)
from src.evaluation import evaluate_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Recommendation System",
    page_icon="🛒",
    layout="wide",
)

# ── Cached data loading ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Load partial dataset for UI speed
    return run_preprocessing_pipeline(nrows=200_000)

@st.cache_resource
def train_all_models(trainset):
    uu = train_user_user_cf(trainset)
    ii = train_item_item_cf(trainset)
    svd = train_svd(trainset)
    return uu, ii, svd

# ── Load & train ──────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models (first run ~30s) …"):
    pipeline_data = load_data()
    df = pipeline_data["df_filtered"]
    trainset = pipeline_data["trainset"]
    testset = pipeline_data["testset"]
    stats = pipeline_data["stats_filtered"]
    uu_model, ii_model, svd_model = train_all_models(trainset)

    uu_results = evaluate_model(uu_model, testset, "User-User CF")
    ii_results = evaluate_model(ii_model, testset, "Item-Item CF")
    svd_results = evaluate_model(svd_model, testset, "SVD")

MODEL_MAP = {
    "User-User CF": uu_model,
    "Item-Item CF": ii_model,
    "SVD (Best)": svd_model,
}
RESULTS_MAP = {
    "User-User CF": uu_results,
    "Item-Item CF": ii_results,
    "SVD (Best)": svd_results,
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    selected_model = st.selectbox("Recommendation Model", list(MODEL_MAP.keys()), index=2)
    n_recs = st.slider("Number of Recommendations", 3, 20, 10)
    st.markdown("---")
    st.metric("Dataset Interactions", f"{stats['n_interactions']:,}")
    st.metric("Unique Users", f"{stats['n_users']:,}")
    st.metric("Unique Products", f"{stats['n_products']:,}")
    st.metric("Sparsity", f"{stats['sparsity']:.1f}%")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🛒 Amazon Electronics Recommendation System")
st.markdown(
    "Compares four recommendation approaches on Amazon Electronics ratings data. "
    "SVD Matrix Factorisation achieves the best RMSE of **0.9039**."
)

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Rating", f"{stats['avg_rating']:.2f} / 5.0")
c2.metric("UU CF RMSE", f"{uu_results['rmse']:.4f}")
c3.metric("II CF RMSE", f"{ii_results['rmse']:.4f}")
c4.metric("SVD RMSE", f"{svd_results['rmse']:.4f}", delta="Best")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dataset Overview", "🔍 EDA", "📊 Model Performance",
    "🎯 Get Recommendations", "📖 How It Works"
])

# ── Tab 1: Dataset Overview ───────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rating Distribution")
        rating_counts = df["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Count"]
        fig = px.bar(
            rating_counts, x="Rating", y="Count",
            color="Count", color_continuous_scale="Blues",
            title="Rating Value Counts"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Dataset Statistics")
        stats_display = {
            "Metric": ["Total Interactions", "Unique Users", "Unique Products",
                       "Sparsity", "Avg Rating", "Median Rating"],
            "Value": [
                f"{stats['n_interactions']:,}", f"{stats['n_users']:,}",
                f"{stats['n_products']:,}", f"{stats['sparsity']:.2f}%",
                f"{stats['avg_rating']:.3f}", f"{stats['median_rating']:.1f}",
            ]
        }
        st.dataframe(pd.DataFrame(stats_display).set_index("Metric"), use_container_width=True)

    st.subheader("Sample Data")
    st.dataframe(df.head(50), use_container_width=True)

# ── Tab 2: EDA ────────────────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Most Active Users")
        top_users = df["userId"].value_counts().head(20).reset_index()
        top_users.columns = ["userId", "Ratings"]
        fig = px.bar(top_users, x="userId", y="Ratings", color="Ratings",
                     color_continuous_scale="Purples", title="Top 20 Active Users")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Most Popular Products")
        top_products = df["productId"].value_counts().head(20).reset_index()
        top_products.columns = ["productId", "Ratings"]
        fig = px.bar(top_products, x="productId", y="Ratings", color="Ratings",
                     color_continuous_scale="Reds", title="Top 20 Popular Products")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ratings per User Distribution")
    user_counts = df["userId"].value_counts()
    fig = px.histogram(
        x=user_counts.values, nbins=50, log_y=True,
        labels={"x": "Number of Ratings", "y": "Number of Users"},
        title="Distribution of Ratings per User (log scale)",
        color_discrete_sequence=["#2ecc71"]
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Model Performance ──────────────────────────────────────────────────
with tab3:
    st.subheader("Model Comparison Table")
    cmp_data = []
    for name, res in RESULTS_MAP.items():
        cmp_data.append({
            "Model": name,
            "RMSE": res["rmse"],
            "Precision@10": res["precision@10"],
            "Recall@10": res["recall@10"],
            "F1@10": res["f1@10"],
        })
    cmp_df = pd.DataFrame(cmp_data).set_index("Model")
    st.dataframe(
        cmp_df.style.highlight_min(subset=["RMSE"], color="#fadbd8")
                    .highlight_max(subset=["Precision@10", "Recall@10", "F1@10"], color="#d5f5e3"),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RMSE Comparison (lower = better)")
        fig = px.bar(
            cmp_df.reset_index(), x="Model", y="RMSE",
            color="RMSE", color_continuous_scale="Reds_r",
            title="RMSE by Model"
        )
        fig.update_yaxes(range=[0.8, cmp_df["RMSE"].max() * 1.1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Precision@10 & Recall@10")
        ranking_df = cmp_df[["Precision@10", "Recall@10"]].reset_index().melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )
        fig = px.bar(ranking_df, x="Model", y="Score", color="Metric",
                     barmode="group", color_discrete_sequence=["#3498db", "#2ecc71"],
                     title="Ranking Metrics by Model")
        fig.update_yaxes(range=[0.7, 1.0])
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Get Recommendations ────────────────────────────────────────────────
with tab4:
    st.subheader("🎯 Personalised Product Recommendations")

    col1, col2 = st.columns([2, 1])
    with col1:
        # Provide sample user IDs
        sample_users = df["userId"].value_counts().index[:10].tolist()
        user_id = st.text_input(
            "Enter User ID",
            value=sample_users[0],
            help="Enter an Amazon user ID from the dataset"
        )
        st.caption(f"Sample user IDs: {', '.join(sample_users[:5])}")
    with col2:
        threshold = st.slider("Minimum Rating Threshold", 1.0, 5.0, 3.5, step=0.5)

    get_btn = st.button("Get Recommendations", type="primary")

    if get_btn or user_id:
        active_algo = MODEL_MAP[selected_model]
        user_history = df[df["userId"] == user_id]

        if user_history.empty:
            st.warning(f"User '{user_id}' not found in the dataset. Try one of the sample IDs above.")
        else:
            col_hist, col_recs = st.columns(2)

            with col_hist:
                st.markdown(f"**Previously Rated Products ({len(user_history)})**")
                history_display = (
                    user_history[["productId", "rating"]]
                    .sort_values("rating", ascending=False)
                    .head(10)
                    .reset_index(drop=True)
                )
                history_display.index += 1
                st.dataframe(history_display, use_container_width=True)

            with col_recs:
                recs = get_recommendations_for_user(
                    active_algo, user_id, df, n=n_recs, threshold=threshold
                )
                st.markdown(f"**Top-{n_recs} Recommendations ({selected_model})**")
                if recs.empty:
                    st.info("No recommendations above threshold. Try lowering the threshold.")
                else:
                    st.dataframe(recs[["rank", "productId", "estimated_rating"]],
                                 use_container_width=True)

    st.divider()
    st.subheader("🏆 Most Popular Products (Rank-Based Baseline)")
    pop_recs = get_rank_based_recommendations(df, min_interactions=50, top_n=10)
    st.dataframe(pop_recs, use_container_width=True)

# ── Tab 5: How It Works ───────────────────────────────────────────────────────
with tab5:
    st.subheader("Algorithm Explanations")

    with st.expander("📊 Rank-Based Recommendation", expanded=False):
        st.markdown("""
**What it does:** Recommends the most popular products based on average rating.

**How:**
1. Compute average rating for each product
2. Filter products with fewer than N ratings (avoids small-sample bias)
3. Return top-N by average rating

**Pros:** Simple, fast, no cold-start problem for products
**Cons:** No personalisation — same recommendations for every user
        """)

    with st.expander("👥 User-User Collaborative Filtering", expanded=False):
        st.markdown("""
**What it does:** Finds users similar to the target user and recommends what they liked.

**How:**
1. Build a user-item matrix
2. Compute cosine similarity between all pairs of users
3. For each target user, find k nearest neighbours
4. Predict ratings based on weighted average of neighbours' ratings

**Best params:** k=60 neighbours, min_k=5, cosine similarity
**RMSE:** 0.9741

**Pros:** Captures diverse user tastes
**Cons:** Computationally expensive for large datasets; suffers from cold-start for new users
        """)

    with st.expander("📦 Item-Item Collaborative Filtering", expanded=False):
        st.markdown("""
**What it does:** Finds products similar to those the user has already rated.

**How:**
1. Build an item-user matrix
2. Compute cosine similarity between products
3. For each product the user hasn't rated, estimate based on similar rated products

**Best params:** k=30 neighbours, min_k=6, cosine similarity
**RMSE:** 0.9752

**Pros:** More stable than User-User (items change less than users); better for new users
**Cons:** Doesn't capture cross-category preferences well
        """)

    with st.expander("🧮 SVD Matrix Factorisation (Best Model)", expanded=True):
        st.markdown("""
**What it does:** Decomposes the user-item rating matrix into latent factor vectors.

**How:**
1. Represent each user as a vector of latent preferences
2. Represent each product as a vector of latent characteristics
3. Predict rating = user_vector · item_vector + biases
4. Train via Stochastic Gradient Descent

**Best params:** n_epochs=30, lr=0.005, regularisation=0.02
**RMSE: 0.9039** ← Best performer

**Pros:** Handles sparsity well; captures complex user-item interactions
**Cons:** Less interpretable; requires retraining when new data arrives
        """)

    st.info(
        "**Why SVD wins:** Unlike neighbourhood methods, SVD learns *why* users like products "
        "(latent factors like 'gaming', 'audio', 'photography') rather than just comparing "
        "surface-level ratings. This generalises better to unseen user-item pairs."
    )
