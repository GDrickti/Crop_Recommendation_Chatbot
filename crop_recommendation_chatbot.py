import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Interactive Crop Recommendation Chatbot")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Step 1: Upload Your Dataset
st.write("### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your agricultural CSV file (with features like soil type, temperature, humidity, crop)", type="csv", key="file_uploader")

# Check if a file is uploaded and can be read
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Verify the dataset has content and columns
        if df.empty:
            st.write("Uploaded file is empty. Please upload a valid CSV file.")
        else:
            # Display dataset preview
            st.write("### Dataset Preview")
            st.write(df.head())

            # Encode target variable if necessary
            if 'crop' in df.columns:
                label_encoder = LabelEncoder()
                df['crop'] = label_encoder.fit_transform(df['crop'])
                
                # Save DataFrame in session state
                st.session_state["df"] = df
                st.session_state["label_encoder"] = label_encoder

                # Dataset Analysis
                st.write("### Data Analysis")
                st.write("#### Crop Distribution")
                crop_counts = df['crop'].value_counts()
                fig, ax = plt.subplots()
                sns.barplot(x=crop_counts.index, y=crop_counts.values, ax=ax)
                ax.set_title("Crop Distribution")
                st.pyplot(fig)

                st.write("#### Feature Correlations")
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.write("Your dataset must include a 'crop' column as the target variable.")
    except Exception as e:
        st.write(f"Error reading file: {e}. Please upload a valid CSV.")
else:
    st.write("Please upload a dataset to continue.")

# Chat Interface
if "df" in st.session_state:
    # Function to append chat messages
    def append_chat(user_message, bot_response):
        st.session_state["chat_history"].append({"user": user_message, "bot": bot_response})

    user_input = st.text_input("You:", key="user_input")

    # Chatbot Responses
    if user_input:
        if "hello" in user_input.lower():
            bot_response = "Hello! Your dataset is loaded. You can ask me to recommend a crop or explore data insights."
            append_chat(user_input, bot_response)

        elif "recommend" in user_input.lower() or "crop" in user_input.lower():
            df = st.session_state["df"]
            label_encoder = st.session_state["label_encoder"]

            # Separate features and target
            feature_columns = [col for col in df.columns if col != 'crop']
            X = df[feature_columns]
            y = df['crop']

            # Train a model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            joblib.dump(model, 'user_crop_model.pkl')

            # Model performance
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"### Model Accuracy: {accuracy:.2f}")
            st.write("#### Classification Report")
            st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

            # Filter out one-hot encoded crop columns from the feature list
            relevant_features = [col for col in feature_columns if not col.startswith("crop_")]

            # Input values for prediction, only showing relevant features
            input_data = {}
            for col in relevant_features:
                input_data[col] = st.number_input(f"Enter {col} value", float(X[col].min()), float(X[col].max()))

            # Get Prediction
            if st.button("Get Recommendation"):
                input_df = pd.DataFrame([input_data])
                prediction_encoded = model.predict(input_df)
                prediction = label_encoder.inverse_transform(prediction_encoded)
                bot_response = f"Recommended Crop: {prediction[0]}"
                append_chat(user_input, bot_response)
        else:
            bot_response = "Please ask for a crop recommendation or say 'hello' to start."
            append_chat(user_input, bot_response)

    # Display Chat History
    for chat in st.session_state["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
