import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# ------------------------------
# Load model and features
# ------------------------------
model = joblib.load("stress_model.pkl")
features = joblib.load("features.pkl")  # ['anxiety_level', 'sleep_quality', 'study_load', ...]
st.set_page_config(page_title="AI Student Stress Dashboard", layout="wide")
st.title("AI Student Stress Dashboard")

# ------------------------------
# Sidebar for user input
# ------------------------------
st.sidebar.header("Student Inputs")

user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.slider(
        label=feature.replace("_", " ").title(),
        min_value=0,
        max_value=3,
        value=1
    )

input_df = pd.DataFrame([user_input])[features]

# ------------------------------
# Stress Prediction
# ------------------------------
stress_pred = model.predict(input_df)[0]
risk_score = float(model.predict_proba(input_df).max() * 100)

# Top contributing factors (largest absolute coefficients)
coef = pd.Series(model.coef_[0], index=features)
top_factors = coef.abs().sort_values(ascending=False).head(3).index.tolist()

# Advice text
advice = (
    f"Your predicted stress level is **{stress_pred}**.\n\n"
    f"The top factors contributing to your stress are **{', '.join(top_factors)}**.\n\n"
    f"Consider improving sleep, managing study load, and reducing peer pressure where possible."
)

# Display prediction
st.subheader("Predicted Stress Level")
st.metric(label="Stress Level", value=stress_pred, delta=f"{risk_score:.2f}%")
st.subheader("Top Contributing Factors & Advice")
st.markdown(advice)

# ------------------------------
# Load dataset for visualizations
# ------------------------------
df = pd.read_csv("StressLevelDataset.csv")
df["stress_label"] = df["stress_level"].map({
    0: "Low Stress",
    1: "Medium Stress",
    2: "High Stress"
})

# ------------------------------
# Stress Distribution Histogram
# ------------------------------
fig1 = px.histogram(
    df,
    x="stress_level",
    color="stress_label",
    nbins=3,
    title="Distribution of Student Stress Levels",
    labels={"stress_level": "Stress Level"},
    text_auto=True
)
fig1.update_layout(template="plotly_white")
# Overlay user stress prediction
pred_val_map = {"Low": 0, "Medium": 1, "High": 2}
fig1.add_scatter(
    x=[pred_val_map.get(stress_pred, 1)],
    y=[0],
    mode="markers",
    marker=dict(size=15, color="black", symbol="x"),
    name="Your Prediction"
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------
# Mental & Support Factors Box Plot
# ------------------------------
mental_cols = ["anxiety_level", "social_support", "future_career_concerns"]
df_long = df.melt(
    id_vars="stress_label",
    value_vars=mental_cols,
    var_name="Feature",
    value_name="Score"
)
fig2 = px.box(
    df_long,
    x="stress_label",
    y="Score",
    color="Feature",
    title="Mental Health & Support Factors Across Stress Levels"
)
# Overlay user input
for feature in mental_cols:
    fig2.add_scatter(
        x=[stress_pred],
        y=[input_df[feature].values[0]],
        mode="markers",
        marker=dict(size=12, color="black", symbol="x"),
        name=f"Your {feature}"
    )
fig2.update_layout(template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# Sleep Quality vs Study Load
# ------------------------------
fig3 = px.scatter(
    df,
    x="sleep_quality",
    y="study_load",
    color="stress_label",
    trendline="ols",
    title="Sleep Quality vs Study Load by Stress Level",
    labels={"sleep_quality": "Sleep Quality", "study_load": "Study Load", "stress_label": "Stress Level"}
)
fig3.add_scatter(
    x=[input_df["sleep_quality"].values[0]],
    y=[input_df["study_load"].values[0]],
    mode="markers",
    marker=dict(size=15, color="black", symbol="x"),
    name="Your Input"
)
fig3.update_layout(template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# Study Load vs Academic Performance
# ------------------------------
fig4 = px.scatter(
    df,
    x="study_load",
    y="academic_performance",
    color="stress_label",
    trendline="ols",
    title="Study Load vs Academic Performance",
    labels={"study_load": "Study Load", "academic_performance": "Academic Performance"}
)
fig4.add_scatter(
    x=[input_df["study_load"].values[0]],
    y=[input_df["academic_performance"].values[0]],
    mode="markers",
    marker=dict(size=15, color="black", symbol="x"),
    name="Your Input"
)
fig4.update_layout(template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------
# Peer Pressure vs Stress
# ------------------------------
fig5 = px.scatter(
    df,
    x="peer_pressure",
    y="stress_level",
    color="stress_label",
    trendline="ols",
    title="Peer Pressure vs Stress Level"
)
fig5.add_scatter(
    x=[input_df["peer_pressure"].values[0]],
    y=[pred_val_map.get(stress_pred, 1)],
    mode="markers",
    marker=dict(size=15, color="black", symbol="x"),
    name="Your Input"
)
fig5.update_layout(template="plotly_white", yaxis=dict(title="Stress Level"))
st.plotly_chart(fig5, use_container_width=True)

# ------------------------------
# Sleep Quality Distribution Box
# ------------------------------
fig6 = px.box(
    df,
    x="stress_label",
    y="sleep_quality",
    color="stress_label",
    title="Sleep Quality Across Stress Levels",
    labels={"stress_label": "Stress Level", "sleep_quality": "Sleep Quality"},
)
fig6.add_scatter(
    x=[stress_pred],
    y=[input_df["sleep_quality"].values[0]],
    mode="markers",
    marker=dict(size=15, color="black", symbol="x"),
    name="Your Input"
)
fig6.update_layout(template="plotly_white")
st.plotly_chart(fig6, use_container_width=True)

# ------------------------------
st.write("Move the sliders in the sidebar to see your stress level update and visualize your input on all graphs!")
