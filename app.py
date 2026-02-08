import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="AI Student Stress Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
 
)

# ------------------------------
# Custom CSS for Modern Styling
# ------------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Background - Darker for better contrast */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30, 60, 114, 0.98) 0%, rgba(126, 34, 206, 0.98) 100%);
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Title Styling - Better contrast */
    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-align: center;
        font-size: 3.5rem !important;
        margin-bottom: 2rem;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.4);
        animation: fadeInDown 0.8s ease-in-out;
        letter-spacing: -1px;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        
    }
    
   
    /* Stress Level Badge - More vibrant and centered */
    .stress-badge {
        display: inline-block;
       
        padding: 25px 50px;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        animation: pulse 2s infinite;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        border: 3px solid rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    .stress-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stress-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stress-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    
    
    /* Sidebar Elements */
    [data-testid="stSidebar"] h2 {
        color: white !important;
        text-align: center;
        margin-bottom: 20px;
        font-size: 1.8rem !important;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    /* Slider styling */
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background: rgba(255, 255, 255, 0.3);
    }
    
    /* Info Box */
    .info-box {
        background: rgba(255, 255, 255, 0.12);
        border-left: 6px solid #fbbf24;
        padding: 25px;
        border-radius: 15px;
        margin: 25px 0;
        color: white !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .info-box h3 {
        color: #fbbf24 !important;
        margin-bottom: 15px;
    }
    
    .info-box strong {
        color: #fbbf24;
    }
    
    /* Better text visibility */
    p, li, div {
        color: rgba(255, 255, 255, 0.95);
    }
    
    /* Metric value styling */
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        margin-top: 40px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load model and features
# ------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("stress_model.pkl")
    features = joblib.load("features.pkl")
    return model, features

@st.cache_data
def load_data():
    df = pd.read_csv("StressLevelDataset.csv")
    df["stress_label"] = df["stress_level"].map({
        0: "Low",
        1: "Medium",
        2: "High"
    })
    return df

model, features = load_model()
df = load_data()

# ------------------------------
# Header
# ------------------------------
st.markdown("<h1> Student Stress Prediction Dashboard</h1>", unsafe_allow_html=True)

# ------------------------------
# Sidebar for user input
# ------------------------------
st.sidebar.markdown("<h2> Student Inputs</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Emoji mapping for features
feature_emojis = {
    "anxiety_level": "",
    "sleep_quality": "",
    "study_load": "",
    "academic_performance": "",
    "peer_pressure": "",
    "social_support": "",
    "future_career_concerns": ""
}

user_input = {}
for feature in features:
    emoji = feature_emojis.get(feature, "📌")
    user_input[feature] = st.sidebar.slider(
        label=f"{emoji} {feature.replace('_', ' ').title()}",
        min_value=0,
        max_value=3,
        value=1,
        key=feature
    )

st.sidebar.markdown("---")
st.sidebar.info(" Adjust the sliders to see real-time predictions!")

input_df = pd.DataFrame([user_input])[features]

# ------------------------------
# Stress Prediction
# ------------------------------
stress_pred = model.predict(input_df)[0]
risk_score = float(model.predict_proba(input_df).max() * 100)

# Top contributing factors
coef = pd.Series(model.coef_[0], index=features)
top_factors = coef.abs().sort_values(ascending=False).head(3)
top_factor_names = [f.replace('_', ' ').title() for f in top_factors.index.tolist()]

# ------------------------------
# Main Dashboard Layout
# ------------------------------

# Row 1: Key Metrics
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    
    # Stress level with color-coded badge
    stress_class = f"stress-{stress_pred.lower()}"
    st.markdown(f"<div class='stress-badge {stress_class}'>{stress_pred}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
   
    # Create a gauge chart for confidence
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 22, 'color': 'white', 'weight': 'bold'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': 'white', 'family': 'Poppins'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': "#fbbf24", 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 3,
            'bordercolor': "rgba(255,255,255,0.3)",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [66, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Poppins"}
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    
    st.markdown("<p style='color: white; font-size:22px; font-weight:bold;'>Top Factors</p>", unsafe_allow_html=True)
    
    # Create horizontal bar chart for top factors with better colors
    colors_map = {
        0: '#10b981',  # Green
        1: '#f59e0b',  # Orange
        2: '#ef4444'   # Red
    }
    
    bar_colors = [colors_map[i] for i in range(len(top_factors))]
    
    fig_factors = go.Figure(go.Bar(
        y=top_factor_names,
        x=top_factors.abs().values,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color='rgba(255, 255, 255, 0.5)', width=2)
        ),
        text=[f'{val:.2f}' for val in top_factors.abs().values],
        textposition='auto',
        textfont=dict(color='white', size=14, family='Poppins', weight='bold')
    ))
    
    fig_factors.update_layout(
    height=250,
    margin=dict(l=210, r=10, t=0, b=20, autoexpand=False),  # Fixed margins
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, tickfont=dict(color='white', size=16, family='Poppins'), automargin=False),
    font={'family': "Poppins", 'size': 16, 'color': 'white'},
    showlegend=False
)
    
    st.plotly_chart(fig_factors, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Row 2: Advice Section

st.markdown(f"""
###  Good Practices for Managing Stress

Your predicted stress level is **{stress_pred}** with **{risk_score:.1f}%** confidence.

**Key Areas to Focus On:**
- **{top_factor_names[0]}**: Primary contributor to your current stress level
- **{top_factor_names[1]}**: Secondary factor affecting your wellbeing
- **{top_factor_names[2]}**: Additional area for potential improvement

**Suggested Actions:**
-  Prioritize 7-9 hours of quality sleep per night
-  Break study sessions into manageable chunks (Pomodoro technique)
-  Strengthen your social support network - reach out to friends and family
- Practice stress-reduction techniques like meditation or exercise
""")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------
# Visualizations Section
# ------------------------------
st.markdown("<h2 style='text-align: center; font-weight:bold;'> Interactive Visualizations</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Prepare prediction value mapping
pred_val_map = {"Low": 0, "Medium": 1, "High": 2}
user_stress_numeric = pred_val_map.get(stress_pred, 1)

# Color scheme - More vibrant
color_discrete_map = {
    "Low": "#10b981",
    "Medium": "#f59e0b", 
    "High": "#ef4444"
}

# Row 3: Distribution Charts
col1, col2 = st.columns(2)

with col1:
   
    
    # Stress Distribution
    fig1 = px.histogram(
        df,
        x="stress_level",
        color="stress_label",
        nbins=3,
        title=" Distribution of Stress Levels",
        labels={"stress_level": "Stress Level", "count": "Number of Students"},
        text_auto=True,
        color_discrete_map=color_discrete_map
    )
    
    fig1.update_layout(
        template="plotly_dark",
        title_font_size=20,
        title_font_color="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='white')),
        height=460,
        margin=dict(l=0, r=40, t=170, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color='white', family='Poppins')
    )
    
    # Add user prediction marker
    fig1.add_scatter(
        x=[user_stress_numeric],
        y=[0],
        mode="markers+text",
        marker=dict(size=25, color="#fbbf24", symbol="star", line=dict(color='white', width=3)),
        name="📍 You",
        text=["YOU"],
        textposition="top center",
        textfont=dict(size=14, color="white", family="Poppins", weight='bold')
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
   
    
    # Radar chart for user profile
    categories = [f.replace('_', ' ').title() for f in features]
    user_values = [user_input[f] for f in features]
    avg_values = [df[f].mean() for f in features]
    
    fig_radar = go.Figure()
    
    # Average student profile
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name='Average Student',
        line_color='rgba(59, 130, 246, 0.8)',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line_width=3
    ))
    
    # User profile
    fig_radar.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Your Profile',
        line_color='#fbbf24',
        fillcolor='rgba(251, 191, 36, 0.3)',
        line_width=3
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 3],
                tickfont=dict(size=10, color='white'),
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='white'),
                gridcolor='rgba(255,255,255,0.2)'
            ),
            bgcolor='rgba(255,255,255,0.05)'
        ),
        showlegend=True,
        title=" Your Profile vs Average Student",
        title_font_size=20,
        title_font_color="white",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color='white')),
        paper_bgcolor='rgba(1,0,0,0)',
        font=dict(color='white', family='Poppins')
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Row 4: Mental Health Factors


mental_cols = ["anxiety_level", "social_support", "future_career_concerns"]
df_long = df.melt(
    id_vars="stress_label",
    value_vars=mental_cols,
    var_name="Feature",
    value_name="Score"
)

df_long["Feature"] = df_long["Feature"].apply(lambda x: x.replace('_', ' ').title())

fig2 = px.box(
    df_long,
    x="stress_label",
    y="Score",
    color="Feature",
    title=" Mental Health & Support Factors Across Stress Levels",
    color_discrete_sequence=['#3b82f6', '#8b5cf6', '#ec4899']
)

# Add user input points
for i, feature in enumerate(mental_cols):
    fig2.add_scatter(
        x=[stress_pred],
        y=[input_df[feature].values[0]],
        mode="markers",
        marker=dict(size=16, color='#fbbf24', symbol="diamond", line=dict(color='white', width=3)),
        name=f"Your {feature.replace('_', ' ').title()}",
        showlegend=(i == 0),
        legendgroup="user"
    )

fig2.update_layout(
    template="plotly_dark",
    title_font_size=20,
    title_font_color="white",
    height=450,
    xaxis_title="Stress Level",
    yaxis_title="Score",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.05)',
    font=dict(color='white', family='Poppins'),
    legend=dict(font=dict(color='white'))
)

st.plotly_chart(fig2, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Row 5: Correlation Charts
col1, col2 = st.columns(2)

with col1:
 
    
    fig3 = px.scatter(
        df,
        x="sleep_quality",
        y="study_load",
        color="stress_label",
        trendline="ols",
        title=" Sleep Quality vs Study Load",
        labels={"sleep_quality": "Sleep Quality", "study_load": "Study Load"},
        color_discrete_map=color_discrete_map,
        opacity=0.7
    )
    
    fig3.add_scatter(
        x=[input_df["sleep_quality"].values[0]],
        y=[input_df["study_load"].values[0]],
        mode="markers",
        marker=dict(size=22, color="#fbbf24", symbol="star", line=dict(color='white', width=3)),
        name=" You"
    )
    
    fig3.update_layout(
        template="plotly_dark",
        title_font_size=20,
        title_font_color="white",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color='white', family='Poppins'),
        legend=dict(font=dict(color='white'))
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:

    
    fig4 = px.scatter(
        df,
        x="study_load",
        y="academic_performance",
        color="stress_label",
        trendline="ols",
        title=" Study Load vs Academic Performance",
        labels={"study_load": "Study Load", "academic_performance": "Academic Performance"},
        color_discrete_map=color_discrete_map,
        opacity=0.7
    )
    
    fig4.add_scatter(
        x=[input_df["study_load"].values[0]],
        y=[input_df["academic_performance"].values[0]],
        mode="markers",
        marker=dict(size=22, color="#fbbf24", symbol="star", line=dict(color='white', width=3)),
        name="You"
    )
    
    fig4.update_layout(
        template="plotly_dark",
        title_font_size=20,
        title_font_color="white",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color='white', family='Poppins'),
        legend=dict(font=dict(color='white'))
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Row 6: Additional Analysis
col1, col2 = st.columns(2)

with col1:
    
    
    fig5 = px.scatter(
        df,
        x="peer_pressure",
        y="stress_level",
        color="stress_label",
        trendline="ols",
        title=" Peer Pressure vs Stress Level",
        color_discrete_map=color_discrete_map,
        opacity=0.7
    )
    
    fig5.add_scatter(
        x=[input_df["peer_pressure"].values[0]],
        y=[user_stress_numeric],
        mode="markers",
        marker=dict(size=22, color="#fbbf24", symbol="star", line=dict(color='white', width=3)),
        name="You"
    )
    
    fig5.update_layout(
        template="plotly_dark",
        title_font_size=20,
        title_font_color="white",
        yaxis=dict(title="Stress Level"),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color='white', family='Poppins'),
        legend=dict(font=dict(color='white'))
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    
    
    fig6 = px.box(
        df,
        x="stress_label",
        y="sleep_quality",
        color="stress_label",
        title=" Sleep Quality Distribution by Stress Level",
        labels={"stress_label": "Stress Level", "sleep_quality": "Sleep Quality"},
        color_discrete_map=color_discrete_map
    )
    
    fig6.add_scatter(
        x=[stress_pred],
        y=[input_df["sleep_quality"].values[0]],
        mode="markers",
        marker=dict(size=22, color="#fbbf24", symbol="star", line=dict(color='white', width=3)),
        name="You"
    )
    
    fig6.update_layout(
        template="plotly_dark",
        title_font_size=20,
        title_font_color="white",
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color='white', family='Poppins')
    )
    
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p style='font-size: 18px; font-weight: 600;'>
         <strong>Pro Tip:</strong> Move the sliders in the sidebar to see real-time updates!
    </p>
    <p style='font-size: 14px; opacity: 0.8; margin-top: 10px;'>
        Built with love using Streamlit & Plotly | AI-Powered Student Wellness Dashboard
    </p>
</div>
""", unsafe_allow_html=True)


