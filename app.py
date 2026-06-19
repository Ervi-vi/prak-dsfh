import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Sistem Prediksi Kekambuhan Kanker Tiroid")

st.write("Masukkan data pasien")

age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=40
)

gender = st.selectbox(
    "Gender",
    encoders["Gender"].classes_
)

smoking = st.selectbox(
    "Smoking",
    encoders["Smoking"].classes_
)

risk = st.selectbox(
    "Risk",
    encoders["Risk"].classes_
)

stage = st.selectbox(
    "Stage",
    encoders["Stage"].classes_
)

response = st.selectbox(
    "Response",
    encoders["Response"].classes_
)

if st.button("Prediksi"):

    data = {
        "Age": age,
        "Gender": encoders["Gender"].transform([gender])[0],
        "Smoking": encoders["Smoking"].transform([smoking])[0],
        "Risk": encoders["Risk"].transform([risk])[0],
        "Stage": encoders["Stage"].transform([stage])[0],
        "Response": encoders["Response"].transform([response])[0]
    }

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Pasien Berpotensi Mengalami Kekambuhan")
    else:
        st.success("Pasien Tidak Mengalami Kekambuhan")
