"""
app.py — Customer Personality Segmentation  ·  Streamlit Dashboard

Launch with:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import run_preprocessing_pipeline, DATA_PATH
from src.clustering import (
    compute_elbow_metrics,
    fit_kmeans,
    assign_clusters,
    get_cluster_profiles,
    get_cluster_summary,
    CLUSTER_NAMES,
)
from src.eda import (
    plot_income_distribution,
    plot_spending_distributions,
    plot_correlation_heatmap,
    plot_campaign_response_rates,
)
from src.visualization import (
    plot_elbow_curve,
    plot_cluster_income_spending,
    plot_cluster_profiles_heatmap,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Personality Segmentation",
    page_icon="👥",
    layout="wide",
)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(n_clusters: int):
    df, X_scaled, scaler = run_preprocessing_pipeline()
    metrics = compute_elbow_metrics(X_scaled, max_k=10)
    model = fit_kmeans(X_scaled, n_clusters=n_clusters)
    df = assign_clusters(df, model, X_scaled)
    return df, X_scaled, model, metrics


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    n_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=5)
    st.markdown("---")
    st.info(
        "Adjust the slider to explore different segmentations. "
        "k=5 was selected as optimal via the Elbow Method."
    )
    st.markdown("---")
    st.markdown("**Project:** MIT Applied Data Science Program")
    st.markdown("**Technique:** K-Means Clustering")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading and segmenting customers …"):
    df, X_scaled, model, elbow_metrics = load_and_preprocess(n_clusters)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("👥 Customer Personality Segmentation")
st.markdown(
    "Interactive dashboard exploring customer segments for targeted marketing. "
    "Built on the **Customer Personality Analysis** dataset (2,240 customers, 29 features)."
)

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{len(df):,}")
c2.metric("Clusters", n_clusters)
c3.metric("Avg Income", f"${df['Income'].mean():,.0f}")
c4.metric("Avg Total Spending", f"${df['TotalSpending'].mean():,.0f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview", "🔍 EDA", "🎯 Segmentation", "📊 Cluster Profiles", "💡 Business Insights"
])

# ── Tab 1: Data Overview ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Preview")
    cols_to_show = ["Income", "Age", "TotalSpending", "TotalPurchases",
                    "TotalCampaigns", "HasChildren", "Cluster_Name"]
    cols_available = [c for c in cols_to_show if c in df.columns]
    st.dataframe(df[cols_available].head(100), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Basic Statistics")
        stat_cols = ["Income", "Age", "TotalSpending", "TotalPurchases"]
        available = [c for c in stat_cols if c in df.columns]
        st.dataframe(df[available].describe().round(2), use_container_width=True)
    with col2:
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values in the processed dataset.")
        else:
            st.dataframe(missing.rename("Count"), use_container_width=True)

# ── Tab 2: EDA ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Income Distribution")
    fig_income = px.histogram(
        df, x="Income", nbins=40, marginal="box",
        color_discrete_sequence=["#3498db"],
        labels={"Income": "Annual Income ($)"},
        title="Customer Income Distribution"
    )
    st.plotly_chart(fig_income, use_container_width=True)

    st.subheader("Spending by Category")
    mnt_cols = [c for c in ["MntWines", "MntFruits", "MntMeatProducts",
                             "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
                if c in df.columns]
    spending_long = df[mnt_cols].melt(var_name="Category", value_name="Amount")
    spending_long["Category"] = spending_long["Category"].str.replace("Mnt", "")
    fig_spend = px.box(
        spending_long, x="Category", y="Amount",
        color="Category", title="Spending Distribution per Category"
    )
    st.plotly_chart(fig_spend, use_container_width=True)

    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number).drop(columns=["Cluster"], errors="ignore")
    corr = num_df.corr()
    fig_corr = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature Correlation Matrix", aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 3: Segmentation ───────────────────────────────────────────────────────
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Elbow Curve")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=elbow_metrics["k_values"], y=elbow_metrics["inertias"],
            mode="lines+markers", name="Inertia", line=dict(color="#3498db")
        ))
        fig_elbow.update_layout(
            xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (WCSS)",
            title="Elbow Method"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        st.subheader("Silhouette Score")
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=elbow_metrics["k_values"], y=elbow_metrics["silhouette_scores"],
            mode="lines+markers", name="Silhouette", line=dict(color="#e74c3c")
        ))
        fig_sil.update_layout(
            xaxis_title="Number of Clusters (k)", yaxis_title="Silhouette Score",
            title="Silhouette Score vs k"
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.subheader("Cluster Distribution")
    counts = df["Cluster_Name"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig_dist = px.bar(
        counts, x="Segment", y="Count", color="Segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"Customer Count per Segment (k={n_clusters})"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Income vs Total Spending")
    fig_scatter = px.scatter(
        df, x="Income", y="TotalSpending", color="Cluster_Name",
        opacity=0.5, size_max=10,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"Income": "Annual Income ($)", "TotalSpending": "Total Spending ($)"},
        title="Income vs Total Spending by Cluster"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Tab 4: Cluster Profiles ───────────────────────────────────────────────────
with tab4:
    st.subheader("Cluster Summary Table")
    summary = get_cluster_summary(df)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Profile Heatmap (Z-Score Normalised)")
    profile_cols = [
        c for c in ["Income", "Age", "TotalSpending", "TotalPurchases",
                    "TotalCampaigns", "NumWebPurchases", "MntWines", "MntMeatProducts"]
        if c in df.columns
    ]
    profile = df.groupby("Cluster_Name")[profile_cols].mean()
    profile_norm = (profile - profile.mean()) / (profile.std() + 1e-9)
    fig_heat = px.imshow(
        profile_norm, color_continuous_scale="RdYlGn", aspect="auto",
        title="Cluster Feature Profiles (Z-Score)", text_auto=".2f"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Spending Patterns by Cluster")
    mnt_cols = [c for c in ["MntWines", "MntFruits", "MntMeatProducts",
                             "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
                if c in df.columns]
    spend_profile = df.groupby("Cluster_Name")[mnt_cols].mean().reset_index()
    spend_long = spend_profile.melt(id_vars="Cluster_Name", var_name="Category", value_name="Avg Spend")
    spend_long["Category"] = spend_long["Category"].str.replace("Mnt", "")
    fig_spend_cluster = px.bar(
        spend_long, x="Category", y="Avg Spend", color="Cluster_Name",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Average Spending per Category by Cluster"
    )
    st.plotly_chart(fig_spend_cluster, use_container_width=True)

# ── Tab 5: Business Insights ──────────────────────────────────────────────────
with tab5:
    st.subheader("Strategic Recommendations per Segment")

    insights = {
        "Premium Heavy Spenders": {
            "emoji": "💎",
            "color": "#2ecc71",
            "desc": "Highest income (~$82k), biggest spenders — especially wine and meat.",
            "actions": [
                "Launch VIP loyalty program with exclusive perks.",
                "Offer premium product bundles (wine + gourmet food).",
                "Invite to exclusive events and early-access campaigns.",
                "High ROI target — invest in retention, not acquisition.",
            ],
        },
        "Affluent Traditionalists": {
            "emoji": "🏬",
            "color": "#3498db",
            "desc": "High income (~$74k), prefer in-store and catalog shopping.",
            "actions": [
                "Enhance in-store experience and loyalty cards.",
                "Send high-quality print catalogs.",
                "Cross-sell wines and meat products.",
                "Personalised email campaigns highlighting new arrivals.",
            ],
        },
        "Digital-Savvy Spenders": {
            "emoji": "💻",
            "color": "#9b59b6",
            "desc": "Mid income (~$57k), web-first shoppers, teenagers at home.",
            "actions": [
                "Optimise website UX and mobile experience.",
                "Push targeted digital ads on social media.",
                "Offer family-friendly deals and bundle promotions.",
                "Leverage web campaigns — highest web purchase volume.",
            ],
        },
        "Low-Income Minimal Spenders": {
            "emoji": "💰",
            "color": "#f39c12",
            "desc": "Lowest income (~$35k), budget-conscious, largest segment.",
            "actions": [
                "Offer value packs and discount promotions.",
                "Loyalty points program to encourage repeat purchases.",
                "Focus on essential product categories (fruit, fish).",
                "Low-cost web acquisition — they prefer digital channels.",
            ],
        },
        "Dormant Families": {
            "emoji": "🔄",
            "color": "#e74c3c",
            "desc": "Recently inactive, many children, smallest segment.",
            "actions": [
                "Win-back email campaign with a compelling discount.",
                "Understand churn reason via survey.",
                "Offer family-oriented promotions.",
                "Consider deprioritising if re-engagement costs outweigh LTV.",
            ],
        },
    }

    for name, info in insights.items():
        with st.expander(f"{info['emoji']} {name}", expanded=False):
            st.markdown(f"**Profile:** {info['desc']}")
            st.markdown("**Recommended Actions:**")
            for action in info["actions"]:
                st.markdown(f"- {action}")
