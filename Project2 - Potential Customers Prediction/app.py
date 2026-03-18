"""
app.py — Lead Conversion Prediction  ·  Streamlit Dashboard

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

from src.preprocessing import run_preprocessing_pipeline, ORDINAL_MAP
from src.models import train_decision_tree, train_random_forest
from src.evaluation import evaluate_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lead Conversion Prediction — ExtraaLearn",
    page_icon="🎓",
    layout="wide",
)

# ── Cached data & models ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return run_preprocessing_pipeline()

@st.cache_resource
def train_models(data):
    dt = train_decision_tree(data["preprocessor"], data["X_train"], data["y_train"])
    rf = train_random_forest(data["preprocessor"], data["X_train"], data["y_train"], tune=False)
    return dt, rf

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models (first run may take ~30s) …"):
    data = load_data()
    dt_model, rf_model = train_models(data)
    dt_metrics = evaluate_model(dt_model, data["X_test"], data["y_test"], "Decision Tree")
    rf_metrics = evaluate_model(rf_model, data["X_test"], data["y_test"], "Random Forest")
    df = data["df"]

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🎓 Lead Conversion Prediction — ExtraaLearn")
st.markdown(
    "Predicts which leads are likely to become paying customers for an EdTech startup, "
    "enabling smarter sales resource allocation."
)

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Leads", f"{len(df):,}")
c2.metric("Conversion Rate", f"{df['status'].mean():.1%}")
c3.metric("DT ROC-AUC", f"{dt_metrics['roc_auc']:.3f}")
c4.metric("RF ROC-AUC", f"{rf_metrics['roc_auc']:.3f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview", "🔍 EDA", "📊 Model Performance",
    "🎯 Lead Scorer", "⭐ Feature Importance"
])

# ── Tab 1: Data Overview ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Conversion Distribution")
        counts = df["status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        counts["Status"] = counts["Status"].map({0: "Not Converted", 1: "Converted"})
        fig = px.pie(counts, values="Count", names="Status",
                     color_discrete_sequence=["#3498db", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Numeric Statistics")
        num_cols = ["age", "website_visits", "time_spent_on_website", "page_views_per_visit"]
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

# ── Tab 2: EDA ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Conversion Rate by Category")
    cat_col = st.selectbox(
        "Select category",
        ["current_occupation", "first_interaction", "last_activity", "profile_completed"]
    )
    rate = df.groupby(cat_col)["status"].mean().reset_index()
    rate.columns = [cat_col, "Conversion Rate"]
    fig = px.bar(
        rate, x=cat_col, y="Conversion Rate", color=cat_col,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"Conversion Rate by {cat_col.replace('_', ' ').title()}"
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric Feature vs Conversion")
    num_col = st.selectbox(
        "Select numeric feature",
        ["age", "time_spent_on_website", "page_views_per_visit", "website_visits"]
    )
    plot_df = df[[num_col, "status"]].copy()
    plot_df["status"] = plot_df["status"].map({0: "Not Converted", 1: "Converted"})
    fig2 = px.box(
        plot_df, x="status", y=num_col, color="status",
        color_discrete_sequence=["#3498db", "#e74c3c"],
        title=f"{num_col.replace('_', ' ').title()} by Conversion Status"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Model Performance ──────────────────────────────────────────────────
with tab3:
    st.subheader("Model Comparison")
    cmp = pd.DataFrame([
        {"Model": "Decision Tree", **{k: v for k, v in dt_metrics.items()
                                      if k in ["accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1"]}},
        {"Model": "Random Forest", **{k: v for k, v in rf_metrics.items()
                                      if k in ["accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1"]}},
    ]).set_index("Model")
    st.dataframe(cmp.style.highlight_max(axis=0, color="#d4efdf"), use_container_width=True)

    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    from sklearn.metrics import roc_curve, roc_auc_score
    for name, model, color in [
        ("Decision Tree", dt_model, "#e74c3c"),
        ("Random Forest", rf_model, "#3498db"),
    ]:
        y_proba = model.predict_proba(data["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(data["y_test"], y_proba)
        auc = roc_auc_score(data["y_test"], y_proba)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})",
            line=dict(color=color, width=2)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="gray")
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        title="ROC Curve Comparison"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# ── Tab 4: Lead Scorer ────────────────────────────────────────────────────────
with tab4:
    st.subheader("🎯 Lead Conversion Probability Scorer")
    st.markdown("Enter lead details below to predict their conversion probability.")

    selected_model_name = st.radio(
        "Model", ["Decision Tree", "Random Forest"], horizontal=True
    )
    active_model = dt_model if selected_model_name == "Decision Tree" else rf_model

    with st.form("lead_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 65, 30)
            occupation = st.selectbox(
                "Current Occupation", ["Professional", "Student", "Unemployed"]
            )
            first_interaction = st.selectbox(
                "First Interaction", ["Website", "Mobile App"]
            )

        with col2:
            profile_completed = st.selectbox(
                "Profile Completed", ["High", "Medium", "Low"]
            )
            website_visits = st.slider("Website Visits", 0, 30, 5)
            time_spent = st.slider("Time Spent on Website (s)", 0, 3000, 500)

        with col3:
            page_views = st.slider("Page Views per Visit", 0.0, 20.0, 3.0, step=0.5)
            last_activity = st.selectbox(
                "Last Activity", ["Website Activity", "Email Activity", "Phone Activity"]
            )
            referral = st.checkbox("Referred by someone")
            digital_media = st.checkbox("Saw digital media ad")
            print_media1 = st.checkbox("Print Media Type 1")
            print_media2 = st.checkbox("Print Media Type 2")
            edu_channels = st.checkbox("Educational Channels")

        submitted = st.form_submit_button("Predict Conversion Probability")

    if submitted:
        lead = pd.DataFrame([{
            "age": age,
            "current_occupation": occupation,
            "first_interaction": first_interaction,
            "profile_completed": profile_completed,
            "profile_completed_ord": ORDINAL_MAP[profile_completed],
            "website_visits": website_visits,
            "time_spent_on_website": time_spent,
            "page_views_per_visit": page_views,
            "last_activity": last_activity,
            "print_media_type1": int(print_media1),
            "print_media_type2": int(print_media2),
            "digital_media": int(digital_media),
            "educational_channels": int(edu_channels),
            "referral": int(referral),
            "is_professional": int(occupation == "Professional"),
        }])

        try:
            proba = active_model.predict_proba(lead)[0][1]
            pred = active_model.predict(lead)[0]

            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                color = "#2ecc71" if proba >= 0.5 else "#e74c3c"
                st.metric(
                    "Conversion Probability",
                    f"{proba:.1%}",
                    delta=f"{'LIKELY TO CONVERT' if pred == 1 else 'UNLIKELY TO CONVERT'}"
                )

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 40], "color": "#fadbd8"},
                            {"range": [40, 70], "color": "#fef9e7"},
                            {"range": [70, 100], "color": "#d5f5e3"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 4}, "value": 50},
                    },
                    title={"text": "Conversion Probability"},
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.subheader("Recommendations")
            if proba >= 0.7:
                st.success("High-priority lead! Assign to senior sales rep immediately.")
            elif proba >= 0.4:
                st.warning("Moderate interest. Nurture with targeted email/phone outreach.")
            else:
                st.error("Low conversion likelihood. Focus on improving profile completion and website engagement.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ── Tab 5: Feature Importance ─────────────────────────────────────────────────
with tab5:
    st.subheader("Top Feature Importances (Random Forest)")
    clf = rf_model.named_steps["classifier"]
    importances = clf.feature_importances_
    try:
        feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)

    fig = px.bar(
        feat_df.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Purples",
        title="Top 15 Feature Importances"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Key Insight:** `time_spent_on_website`, `first_interaction`, and `profile_completed` "
        "are the strongest predictors of lead conversion. Focus sales efforts on leads with "
        "high website engagement and complete profiles."
    )
