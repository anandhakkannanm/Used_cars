import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.model_selection import train_test_split

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# --- DATA LOADING AND PREPROCESSING ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Car details v3.csv")
    df = df.dropna(subset=['selling_price'])
    
    df['mileage'] = df['mileage'].astype(str).str.extract(r'(\d+\.\d+)').astype(float)
    df['engine'] = df['engine'].astype(str).str.extract(r'(\d+)').astype(float)
    df['max_power'] = df['max_power'].astype(str).str.extract(r'(\d+\.\d+)').astype(float)
    
    df['seats'] = df['seats'].fillna(df['seats'].mode()[0])
    df['brand'] = df['name'].str.split(' ').str[0]
    
    df = df.drop(columns=['torque', 'name'], errors='ignore')
    df = df.dropna()
    
    return df

@st.cache_resource
def train_model(df):
    X = df[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats', 'brand']]
    y = df['selling_price']
    
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
    numerical_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)
    return model

try:
    df = load_and_clean_data()
    model = train_model(df)
except FileNotFoundError:
    st.error("Dataset 'Car details v3.csv' not found. Please ensure it is in the same directory as app.py")
    st.stop()

# --- STREAMLIT UI ---
st.title("🚗 Used Car Price Predictor")
st.markdown("Enter the details of the car below to get an estimated selling price.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Car Details")
    brands = sorted(df['brand'].unique())
    selected_brand = st.selectbox("Brand", brands)
    year = st.slider("Manufacturing Year", int(df['year'].min()), int(df['year'].max()), 2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
    fuel_type = st.selectbox("Fuel Type", df['fuel'].unique())

with col2:
    st.subheader("Transmission & Seller")
    seller_type = st.selectbox("Seller Type", df['seller_type'].unique())
    transmission = st.selectbox("Transmission", df['transmission'].unique())
    owner_type = st.selectbox("Owner Type", df['owner'].unique())
    seats = st.selectbox("Seats", sorted(df['seats'].unique()))

with col3:
    st.subheader("Specifications")
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=18.0, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200, step=100)
    max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=400.0, value=80.0, step=1.0)

if st.button("Predict Selling Price", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel_type],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner_type],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'brand': [selected_brand]
    })
    
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹ {predicted_price:,.2f}")
