import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests

st.title("Interactive Crop Recommendation Chatbot")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to append chat messages
def append_chat(user_message, bot_response):
    st.session_state["chat_history"].append({"user": user_message, "bot": bot_response})

# Chat Interface
st.write("### Chat with the Crop Recommendation Bot")
user_input = st.text_input("You:", key="user_input")

# File upload handling
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# Chatbot Responses
if user_input:
    # Greeting and basic responses
    if "hello" in user_input.lower():
        bot_response = "Hello! Please upload your agricultural dataset, and I’ll help recommend a crop for you."
        append_chat(user_input, bot_response)
        
    elif "upload" in user_input.lower():
        # Allow file upload
        st.session_state["uploaded_file"] = st.file_uploader("Upload your CSV file", type="csv")
        bot_response = "Upload your dataset here. It should contain features like soil type, temperature, and crop."
        append_chat(user_input, bot_response)
        
    # Check if dataset is uploaded
    elif st.session_state["uploaded_file"]:
        df = pd.read_csv(st.session_state["uploaded_file"])
        
        # If user requests a recommendation
        if "recommend" in user_input.lower() or "crop" in user_input.lower():
            if 'crop' not in df.columns:
                bot_response = "Your dataset must include a 'crop' column as the target variable."
                append_chat(user_input, bot_response)
            else:
                feature_columns = [col for col in df.columns if col != 'crop']
                X = df[feature_columns]
                y = df['crop']
                
                # Train a model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                joblib.dump(model, 'user_crop_model.pkl')
                
                # Prompt for input data
                bot_response = "Your model is trained! Now, enter data for a crop recommendation."
                append_chat(user_input, bot_response)
                
                # Prompt user for each feature
                input_data = {}
                for col in feature_columns:
                    input_data[col] = st.number_input(f"Enter {col} value", float(df[col].min()), float(df[col].max()))
                
                # Get Prediction
                if st.button("Get Recommendation"):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)
                    bot_response = f"Recommended Crop: {prediction[0]}"
                    append_chat(user_input, bot_response)
        else:
            bot_response = "Please ask for a crop recommendation or upload a new dataset."
            append_chat(user_input, bot_response)
    
    # If no file uploaded
    else:
        bot_response = "Please upload a dataset first."
        append_chat(user_input, bot_response)

# Display Chat History
for chat in st.session_state["chat_history"]:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**Bot:** {chat['bot']}")
