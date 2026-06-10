import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Saved Files
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="Credit Card Fraud Detection")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and predict whether a transaction is fraudulent.")

# -----------------------------
# User Inputs
# -----------------------------

merchant = st.selectbox("Merchant", encoders["merchant"].classes_)
category = st.selectbox(
    "Category",
    encoders["category"].classes_
)

amt = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

gender = st.selectbox(
    "Gender",
    encoders["gender"].classes_
)

city = st.selectbox(
    "City",
    encoders["city"].classes_
)

state = st.selectbox(
    "State",
    encoders["state"].classes_
)
zip_code = st.number_input(
    "ZIP Code",
    min_value=0,
    step=1
)

lat = st.number_input(
    "Customer Latitude",
    value=0.0,
    format="%.6f"
)

long = st.number_input(
    "Customer Longitude",
    value=0.0,
    format="%.6f"
)

city_pop = st.number_input(
    "City Population",
    min_value=0,
    step=1
)
job = st.selectbox(
    "Job",
    encoders["job"].classes_
)

unix_time = st.number_input(
    "Unix Time",
    min_value=0,
    step=1
)

merch_lat = st.number_input(
    "Merchant Latitude",
    value=0.0,
    format="%.6f"
)

merch_long = st.number_input(
    "Merchant Longitude",
    value=0.0,
    format="%.6f"
)

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2100,
    value=2024
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=1
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=1
)

hour = st.number_input(
    "Hour",
    min_value=0,
    max_value=23,
    value=12
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=120,
    value=30
)

night_transaction = st.selectbox(
    "Night Transaction",
    [0, 1]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    try:

        merchant = encoders["merchant"].transform([merchant])[0]
        category = encoders["category"].transform([category])[0]
        gender = encoders["gender"].transform([gender])[0]
        city = encoders["city"].transform([city])[0]
        state = encoders["state"].transform([state])[0]
        job = encoders["job"].transform([job])[0]

        input_df = pd.DataFrame([{
            "merchant": merchant,
            "category": category,
            "amt": amt,
            "gender": gender,
            "city": city,
            "state": state,
            "zip": zip_code,
            "lat": lat,
            "long": long,
            "city_pop": city_pop,
            "job": job,
            "unix_time": unix_time,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "age": age,
            "night_transaction": night_transaction
        }])

        scale_cols = [
            "amt",
            "city_pop",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "age"
        ]

        input_df[scale_cols] = scaler.transform(
            input_df[scale_cols]
        )

        prediction = model.predict(input_df)

        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

    except ValueError as e:

        st.warning(
            "Entered category/merchant/city/state/job was not seen during training.\n\n"
            f"Details: {e}"
        )

    except Exception as e:

        st.error(f"Error: {e}")