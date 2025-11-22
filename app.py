
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# --- 1. Load Trained Model and Preprocessing Objects ---

@st.cache_resource
def load_artifacts():
    try:
        stacked_model = joblib.load('stacked_model.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        scaler_kmeans = joblib.load('scaler_kmeans.pkl')
        kmeans = joblib.load('kmeans.pkl')
        scaler_numerical = joblib.load('scaler_numerical.pkl')
        with open('feature_col.json', 'r') as f:
            feature_columns = json.load(f)
        return stacked_model, target_encoder, scaler_kmeans, kmeans, scaler_numerical, feature_columns
    except Exception as e:
        st.error(f"Error loading model or preprocessing objects: {e}")
        st.stop()

stacked_model, target_encoder, scaler_kmeans, kmeans, scaler_numerical, feature_columns = load_artifacts()

# --- 2. Define Constants for Preprocessing (derived from training) ---
# These were determined during the preprocessing phase in the notebook
all_amenities = ['Playground', 'Gym', 'Garden', 'Pool', 'Clubhouse'] # Explicitly defined
clustering_features = ['Size_in_SqFt', 'BHK', 'Nearby_Schools', 'Nearby_Hospitals', 'Age_of_Property', 'Year_Built', 'Floor_No', 'Total_Floors'] # Explicitly defined

# --- 3. Preprocessing Function for New Input Data ---
def preprocess_input(input_df):
    processed_df = input_df.copy()

    # a. Handle 'Amenities' feature: create boolean columns
    for amenity in all_amenities:
        col_name = f'has_{amenity.lower().replace(" ", "_")}'
        processed_df[col_name] = processed_df['Amenities'].apply(lambda x: 1 if pd.notna(x) and amenity in x else 0)
    processed_df = processed_df.drop(columns=['Amenities'])

    # b. Apply previously fitted target_encoder to 'Locality'
    processed_df['Locality'] = target_encoder.transform(processed_df['Locality'])

    # c. Recreate the 'cluster' feature using previously fitted scaler_kmeans and kmeans model
    input_clustering = processed_df[clustering_features]
    input_scaled_clustering = scaler_kmeans.transform(input_clustering)
    processed_df['cluster'] = kmeans.predict(input_scaled_clustering)

    # d. Perform one-hot encoding on other categorical features
    categorical_cols_ohe_input = processed_df.select_dtypes(include=['object']).columns.tolist()
    input_encoded = pd.get_dummies(processed_df[categorical_cols_ohe_input], drop_first=True)

    processed_df = pd.concat([processed_df.drop(columns=categorical_cols_ohe_input), input_encoded], axis=1)

    # Ensure the resulting columns match the X_train DataFrame exactly
    # Add missing columns (from X_train.columns but not in processed_df) with value 0
    missing_cols = set(feature_columns) - set(processed_df.columns)
    for c in missing_cols:
        processed_df[c] = 0
    # Drop extra columns (in processed_df but not in X_train.columns)
    extra_cols = set(processed_df.columns) - set(feature_columns)
    processed_df = processed_df.drop(columns=list(extra_cols))
    
    # Reorder columns to match the training data's order
    processed_df = processed_df[feature_columns]

    # e. Apply previously fitted scaler_numerical to all numerical features
    scaled_cols_for_prediction = [col for col in feature_columns if col in ['Locality', 'BHK', 'Size_in_SqFt', 'Price_per_SqFt', 'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals', 'has_playground', 'has_gym', 'has_garden', 'has_pool', 'has_clubhouse']]
    
    for col in scaled_cols_for_prediction:
        if col not in processed_df.columns:
            processed_df[col] = 0 

    processed_df[scaled_cols_for_prediction] = scaler_numerical.transform(processed_df[scaled_cols_for_prediction])

    return processed_df

# --- 4. Streamlit Application Layout ---
st.set_page_config(layout="wide", page_title="House Price Estimator")
st.title("🏡 India House Price Estimator")
st.markdown("--- Developed using a Stacked Ensemble Model ---

Enter the property details below to get an estimated price in Lakhs.")

# Input Form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        State = st.selectbox("State", ['Tamil Nadu', 'Maharashtra', 'Punjab', 'Rajasthan', 'West Bengal', 'Karnataka', 'Uttar Pradesh', 'Gujarat', 'Madhya Pradesh', 'Haryana', 'Kerala', 'Telangana', 'Andhra Pradesh', 'Bihar', 'Odisha', 'Chhattisgarh', 'Assam', 'Jharkhand', 'Uttarakhand'])
        City = st.text_input("City (e.g., Chennai, Pune)")
        Locality = st.text_input("Locality (e.g., Locality_84, Locality_490)")
        Property_Type = st.selectbox("Property Type", ['Apartment', 'Independent House', 'Villa', 'Penthouse', 'Studio Apartment'])
        BHK = st.number_input("BHK (Bedrooms, Hall, Kitchen)", min_value=1, max_value=10, value=2)
        Size_in_SqFt = st.number_input("Size in SqFt", min_value=100.0, max_value=20000.0, value=1200.0)
        Price_per_SqFt = st.number_input("Price per SqFt (approx)", min_value=0.01, max_value=1.0, value=0.08, format="%.2f")
    with col2:
        Year_Built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2010)
        Floor_No = st.number_input("Floor Number", min_value=0, max_value=100, value=5)
        Total_Floors = st.number_input("Total Floors in Building", min_value=1, max_value=100, value=10)
        Age_of_Property = st.number_input("Age of Property (Years)", min_value=0, max_value=100, value=14)
        Nearby_Schools = st.number_input("Number of Nearby Schools (0-10)", min_value=0, max_value=10, value=5)
        Nearby_Hospitals = st.number_input("Number of Nearby Hospitals (0-10)", min_value=0, max_value=10, value=3)
        Public_Transport_Accessibility = st.selectbox("Public Transport Accessibility", ['High', 'Medium', 'Low'])

    with col3:
        Parking_Space = st.selectbox("Parking Space Available?", ['Yes', 'No'])
        Security = st.selectbox("Security Available?", ['Yes', 'No'])
        Furnished_Status = st.selectbox("Furnished Status", ['Furnished', 'Semi-Furnished', 'Unfurnished'])
        Facing = st.selectbox("Facing Direction", ['North', 'East', 'South', 'West', 'North-East', 'North-West', 'South-East', 'South-West'])
        Owner_Type = st.selectbox("Owner Type", ['Owner', 'Builder', 'Broker'])
        Availability_Status = st.selectbox("Availability Status", ['Ready_to_Move', 'Under_Construction'])
        amenities_input = st.multiselect("Amenities (Select all that apply)", all_amenities)
        Amenities = ", ".join(amenities_input) if amenities_input else "None"

    submitted = st.form_submit_button("Estimate Price")

    if submitted:
        input_data = pd.DataFrame({
            'State': [State],
            'City': [City],
            'Locality': [Locality],
            'Property_Type': [Property_Type],
            'BHK': [BHK],
            'Size_in_SqFt': [Size_in_SqFt],
            'Price_per_SqFt': [Price_per_SqFt],
            'Year_Built': [Year_Built],
            'Floor_No': [Floor_No],
            'Total_Floors': [Total_Floors],
            'Age_of_Property': [Age_of_Property],
            'Nearby_Schools': [Nearby_Schools],
            'Nearby_Hospitals': [Nearby_Hospitals],
            'Public_Transport_Accessibility': [Public_Transport_Accessibility],
            'Parking_Space': [Parking_Space],
            'Security': [Security],
            'Furnished_Status': [Furnished_Status],
            'Facing': [Facing],
            'Owner_Type': [Owner_Type],
            'Availability_Status': [Availability_Status],
            'Amenities': [Amenities]
        })

        processed_input_data = preprocess_input(input_data)
        predicted_price = stacked_model.predict(processed_input_data)[0]

        st.success(f"### Estimated House Price: ₹ {predicted_price:.2f} Lakhs")
        st.info("Disclaimer: This is an estimation based on the trained model and may not reflect the exact market price.")
