import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Interactive Crop Recommendation Chatbot")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Step 1: Upload Dataset
st.write("### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your agricultural CSV file (with features like soil type, temperature, crop)", type="csv")

# Load and preview dataset
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.write("Uploaded file is empty. Please upload a valid CSV file.")
        else:
            st.session_state["df"] = df
            st.write("### Dataset Preview")
            st.write(df.head())
    except Exception as e:
        st.write(f"Error reading file: {e}. Please upload a valid CSV.")
else:
    st.write("Please upload a dataset to continue.")

# Define function to append messages to chat history
def append_chat(user_message, bot_response):
    st.session_state["chat_history"].append({"user": user_message, "bot": bot_response})

# Chat Interface with Enhanced Chat Flow
if "df" in st.session_state:
    user_input = st.text_input("You:", key="user_input")

    if user_input:
        df = st.session_state["df"]
        # Response to Greeting
        if "hello" in user_input.lower():
            bot_response = "Hello! I'm here to assist you with crop recommendations and dataset analysis."
            append_chat(user_input, bot_response)

        # Provide Dataset Insights
        elif "data insights" in user_input.lower() or "data analysis" in user_input.lower():
            bot_response = "Here are some insights from your dataset:"
            append_chat(user_input, bot_response)

            st.write("### Basic Statistics")
            st.write(df.describe())
            st.write("### Crop Distribution")
            st.bar_chart(df['crop'].value_counts())

        # Crop Recommendation
        elif "recommend" in user_input.lower() or "crop" in user_input.lower():
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
                accuracy = accuracy_score(y_test, model.predict(X_test))

                bot_response = f"Model trained with accuracy: {accuracy:.2f}. Now, please enter data for a crop recommendation."
                append_chat(user_input, bot_response)

                # Prompt user for each feature
                input_data = {}
                for col in feature_columns:
                    input_data[col] = st.number_input(f"Enter {col} value", float(X[col].min()), float(X[col].max()))

                # Get Prediction
                if st.button("Get Recommendation"):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)
                    bot_response = f"Recommended Crop: {prediction[0]}"
                    append_chat(user_input, bot_response)

        # Retrain the Model with Parameters
        elif "retrain" in user_input.lower() or "train" in user_input.lower():
            bot_response = "Specify parameters for model retraining (RandomForestClassifier)!"
            append_chat(user_input, bot_response)

            n_estimators = st.slider("Number of Estimators", 50, 300, step=10, value=100)
            max_depth = st.slider("Max Depth", 5, 50, step=5, value=20)

            if st.button("Retrain Model"):
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                joblib.dump(model, 'user_crop_model.pkl')
                new_accuracy = accuracy_score(y_test, model.predict(X_test))
                
                bot_response = f"Model retrained with new accuracy: {new_accuracy:.2f}."
                append_chat(user_input, bot_response)

        else:
            bot_response = "I'm here to provide crop recommendations, data insights, or retrain the model. Please specify your request."
            append_chat(user_input, bot_response)

    # Display Chat History
    for chat in st.session_state["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
