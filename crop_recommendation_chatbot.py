import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Interactive Crop Recommendation Chatbot")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to append chat messages
def append_chat(user_message, bot_response):
    st.session_state["chat_history"].append({"user": user_message, "bot": bot_response})

# Prompt for dataset upload at the beginning
uploaded_file = st.file_uploader("Please upload your agricultural CSV file (e.g., with features like soil type, temperature, etc.)", type="csv")

if uploaded_file:
    st.session_state["uploaded_file"] = uploaded_file
    df = pd.read_csv(st.session_state["uploaded_file"])
    st.write("### Dataset Preview")
    st.write(df.head())
    
    if "crop" in df.columns:
        feature_columns = [col for col in df.columns if col != 'crop']
        X = df[feature_columns]
        y = df['crop']
        
        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, 'user_crop_model.pkl')
        
        st.write("Your model is trained! Now enter data for a crop recommendation.")
        
        # Collect input for crop recommendation
        input_data = {}
        for col in feature_columns:
            input_data[col] = st.number_input(f"Enter {col} value", float(df[col].min()), float(df[col].max()))
        
        if st.button("Get Recommendation"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            bot_response = f"Recommended Crop: {prediction[0]}"
            append_chat("Crop recommendation request", bot_response)
            st.write(f"**Bot:** {bot_response}")
    else:
        st.error("Dataset must include a 'crop' column as the target variable.")
else:
    st.write("Please upload your dataset first.")
