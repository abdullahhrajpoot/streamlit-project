"""
Heart Disease Prediction System
Abdullah Hassan | FA24-BSE-007
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="heart", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]


model, scaler = load_model()

st.title("Heart Disease Prediction System")
st.markdown("**Abdullah Hassan | FA24-BSE-007**")
st.markdown(
    """
    Enter patient details below to predict the presence of heart disease.
    The model uses a **Logistic Regression** classifier trained on the
    UCI Heart Disease dataset (processed Cleveland, 303 patients).

    *Model accuracy on test data: **77%***
    """
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
    cp = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Typical Angina",
            2: "2 - Atypical Angina",
            3: "3 - Non-Anginal Pain",
            4: "4 - Asymptomatic",
        }[x],
        index=3,
    )

with col2:
    sex = st.selectbox("Sex", options=["Male", "Female"])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=140)

if st.button("Predict", type="primary", use_container_width=True):
    sex_encoded = 1 if sex == "Male" else 0

    le_cp = LabelEncoder().fit([1, 2, 3, 4])
    cp_encoded = le_cp.transform([cp])[0]

    input_df = pd.DataFrame(
        {"age": [age], "sex": [sex_encoded], "cp": [cp_encoded], "thalach": [thalach]}
    )

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.error("### Heart Disease Detected")
    else:
        st.success("### No Heart Disease")

    with st.expander("See the input feature vector"):
        st.dataframe(input_df, hide_index=True)

st.divider()
st.caption("Abdullah Hassan | FA24-BSE-007")
