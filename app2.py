!pipinstalljoblib


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved files
model = joblib.load("house_price_knn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 House Price Prediction")

# User inputs
area = st.slider("Area", 500, 5000, 1500)
bedrooms = st.slider("Bedrooms", 1, 6, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)
floors = st.slider("Floors", 1, 5, 2)
year_built = st.slider("Year Built", 1900, 2025, 2000)

location = st.selectbox(
    "Location",
    ["Downtown", "Suburban", "Rural"]
)

condition = st.selectbox(
    "Condition",
    ["Excellent", "Good", "Fair", "Poor"]
)

garage = st.selectbox(
    "Garage",
    ["Yes", "No"]
)

if st.button("Predict Price"):

    # Create a dataframe with the numeric inputs
    input_df = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Floors": [floors],
        "YearBuilt": [year_built],
        "Location": [location],
        "Condition": [condition],
        "Garage": [garage]
    })

    # Apply the same one-hot encoding used during training
    input_df = pd.get_dummies(input_df)

    # Add any missing columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated House Price: ${prediction:,.2f}")






