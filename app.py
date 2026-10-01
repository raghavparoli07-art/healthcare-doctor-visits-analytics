import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Doctor Visit Predictor", layout="wide")
model = joblib.load(BASE / "model.pkl")

st.title("Doctor Visit Predictor")
st.caption("Predicts whether a person is likely to visit a doctor at least once, based on health, income and insurance factors.")

yn = ["no", "yes"]
with st.sidebar:
    st.header("Patient features")
    gender = st.selectbox("Gender", ["female", "male"])
    age = st.slider("Age (years ÷ 100)", 0.19, 0.72, 0.40, 0.01, help="0.40 = 40 years")
    income = st.slider("Annual income (in $10,000s)", 0.0, 1.5, 0.55, 0.01)
    illness = st.slider("Illnesses in past 2 weeks", 0, 5, 1)
    reduced = st.slider("Days of reduced activity (past 2 weeks)", 0, 14, 0)
    health = st.slider("General health questionnaire score (higher = worse)", 0, 12, 0)
    private = st.selectbox("Private insurance", yn)
    freepoor = st.selectbox("Free government cover (low income)", yn)
    freerepat = st.selectbox("Free government cover (repatriation)", yn)
    nchronic = st.selectbox("Chronic condition (not limiting activity)", yn)
    lchronic = st.selectbox("Chronic condition (limiting activity)", yn)

row = pd.DataFrame([{"gender": gender, "age": age, "income": income, "illness": illness,
                     "reduced": reduced, "health": health, "private": private,
                     "freepoor": freepoor, "freerepat": freerepat,
                     "nchronic": nchronic, "lchronic": lchronic}])
probability = model.predict_proba(row)[0, 1]

c1, c2 = st.columns(2)
c1.metric("Likely to visit doctor?", "Yes" if probability >= 0.5 else "No")
c2.metric("Probability of visiting", f"{probability:.1%}")
st.progress(float(probability))

names = ["visits_dist", "mean_visits_by_group", "illness_reduced", "corr", "feat_importance", "confusion"]
tabs = st.tabs(["Visit counts", "Group means", "Illness and reduced", "Correlations", "Feature importance", "Confusion matrix"])
for tab, name in zip(tabs, names):
    with tab:
        st.image(str(BASE / "out" / f"{name}.png"), width="stretch")
