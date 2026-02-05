# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# -------------------------
# Input Schema
# -------------------------
class StudentData(BaseModel):
    anxiety_level: int
    depression: int
    self_esteem: int
    mental_health_history: int
    headache: int
    blood_pressure: int
    sleep_quality: int
    breathing_problem: int
    noise_level: int
    living_conditions: int
    safety: int
    basic_needs: int
    academic_performance: int
    study_load: int
    teacher_student_relationship: int
    future_career_concerns: int
    social_support: int
    peer_pressure: int
    extracurricular_activities: int
    bullying: int

# -------------------------
# Initialize App & Load Model
# -------------------------
app = FastAPI(title="AI Stress Predictor")

# Load ML model & features
model = joblib.load("stress_model.pkl")
features = joblib.load("features.pkl")

# -------------------------
# Home Route
# -------------------------
@app.get("/")
def home():
    return {"message": "AI Stress Predictor API is running"}

# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict-stress")
def predict_stress(data: StudentData):
    try:
        # Convert input to dict and fill missing features
        input_dict = data.dict()
        for feat in features:
            if feat not in input_dict:
                input_dict[feat] = 0

        # Create DataFrame in correct order
        input_df = pd.DataFrame([input_dict])[features]

        # Predict stress level (string labels: 'High', 'Low', 'Medium')
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # Compute top 3 stress factors safely
        try:
            coefs = model.coef_[0]
            impact_df = pd.DataFrame({
                "feature": features,
                "impact": coefs * input_df.iloc[0].values
            }).sort_values(by="impact", key=abs, ascending=False)
            top_factors = impact_df["feature"].head(3).tolist()
        except Exception:
            top_factors = []

        # Dynamic GenAI-style advice per factor
        advice_parts = []
        for factor in top_factors:
            value = input_dict.get(factor, 0)
            if "sleep" in factor:
                advice_parts.append(f"Improve your {factor.replace('_',' ')} to reduce stress.")
            elif "study" in factor or "academic" in factor:
                advice_parts.append(f"Manage your {factor.replace('_',' ')} for better balance.")
            elif "anxiety" in factor or "depression" in factor:
                advice_parts.append(f"Practice mindfulness to lower {factor.replace('_',' ')}.")
            elif "social_support" in factor or "peer" in factor:
                advice_parts.append(f"Engage with supportive friends to improve {factor.replace('_',' ')}.")
            else:
                advice_parts.append(f"Work on {factor.replace('_',' ')} to reduce stress.")

        advice_text = (
            f"Your predicted stress level is {prediction}. "
            f"The top factors contributing to your stress are {', '.join(top_factors) if top_factors else 'not available'}. "
            + " ".join(advice_parts)
        )

        return {
            "stress_level": prediction,
            "risk_score": round(max(proba) * 100, 2),
            "top_factors": top_factors,
            "advice": advice_text
        }

    except Exception as e:
        return {"error": str(e)}
