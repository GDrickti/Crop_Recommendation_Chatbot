import streamlit as st
pip install joblib
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests

# Title
st.title("Custom Crop Recommendation Chatbot")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your agricultural CSV file", type="csv")

if uploaded_file:
    # Step 2: Load and display the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Step 3: Model Training (If no pre-trained model exists)
    if 'crop' not in df.columns:
        st.error("Dataset must include a 'crop' column as the target variable.")
    else:
        # Select feature columns
        feature_columns = [col for col in df.columns if col != 'crop']
        
        # Split the data
        X = df[feature_columns]
        y = df['crop']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a RandomForest model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, 'user_crop_model.pkl')
        st.success("Model trained successfully!")

        # Prompt user for input data for recommendation
        st.write("## Provide data for crop recommendation")
        input_data = {}
        for col in feature_columns:
            input_data[col] = st.number_input(f"Enter {col} value", float(df[col].min()), float(df[col].max()))

        if st.button("Get Crop Recommendation"):
            # Make Prediction
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.success(f"Recommended Crop: {prediction[0]}")
