import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_knn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 House Price Prediction")

area = st.slider("Area", 500, 5000, 1500)
bedrooms = st.slider("Bedrooms", 1, 6, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)
floors = st.slider("Floors", 1, 5, 2)
year_built = st.slider("Year Built", 1900, 2025, 2000)

# These options MUST match the model
location = st.selectbox(
    "Location",
    ["Urban", "Suburban", "Rural"]
)

condition = st.selectbox(
    "Condition",
    ["Excellent", "Good", "Fair", "Poor"]
)

garage = st.selectbox(
    "Garage",
    ["No", "Yes"]
)

if st.button("Predict Price"):

    # Start with all features set to 0
    input_data = {col: 0 for col in columns}

    # Numeric features
    input_data["Area"] = area
    input_data["Bedrooms"] = bedrooms
    input_data["Bathrooms"] = bathrooms
    input_data["Floors"] = floors
    input_data["YearBuilt"] = year_built

    # Location
    if location != "Urban":
        input_data[f"Location_{location}"] = 1

    # Condition
    if condition != "Excellent":
        input_data[f"Condition_{condition}"] = 1

    # Garage
    if garage == "Yes":
        input_data["Garage_Yes"] = 1

    input_df = pd.DataFrame([input_data])

    # Ensure correct column order
    input_df = input_df[columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated House Price: ${prediction:,.2f}")


