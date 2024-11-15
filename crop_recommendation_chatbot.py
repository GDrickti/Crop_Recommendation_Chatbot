import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

st.title("Interactive Crop Recommendation Chatbot")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to append chat messages
def append_chat(user_message, bot_response):
    st.session_state["chat_history"].append({"user": user_message, "bot": bot_response})

# Dynamic Greeting
current_hour = datetime.now().hour
if 5 <= current_hour < 12:
    greeting = "Good Morning"
elif 12 <= current_hour < 18:
    greeting = "Good Afternoon"
else:
    greeting = "Good Evening"

# Step 1: Upload Your Dataset
st.write("### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader(
    "Upload your agricultural CSV file (with features like soil type, temperature, crop)", 
    type="csv", 
    key="unique_uploader_key"
)

# Check if a file is uploaded and can be read
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

# Chat Interface
if "df" in st.session_state:
    user_input = st.text_input("You:", key="user_input")

    if user_input:
        if "hello" in user_input.lower():
            bot_response = f"{greeting}! Your dataset is loaded. How can I assist you? You can ask for crop recommendations or data insights."
            append_chat(user_input, bot_response)

        elif "data description" in user_input.lower():
            num_rows, num_cols = st.session_state["df"].shape
            bot_response = f"Dataset Description:\n- Number of rows: {num_rows}\n- Number of columns: {num_cols}"
            append_chat(user_input, bot_response)
            st.write("### Data Description")
            st.write(f"Number of rows: {num_rows}")
            st.write(f"Number of columns: {num_cols}")
            st.write(st.session_state["df"].dtypes)

        elif "data summary" in user_input.lower():
            bot_response = "Here's the summary of the dataset."
            append_chat(user_input, bot_response)
            st.write("### Dataset Summary")
            st.write(st.session_state["df"].describe(include='all'))

        elif "data analysis" in user_input.lower():
            bot_response = "Performing data analysis for each column..."
            append_chat(user_input, bot_response)
            for column in st.session_state["df"].columns:
                st.write(f"#### Analysis of `{column}`")
                if pd.api.types.is_numeric_dtype(st.session_state["df"][column]):
                    st.write(st.session_state["df"][column].describe())
                    st.write("Distribution")
                    fig, ax = plt.subplots()
                    sns.histplot(st.session_state["df"][column], kde=True, ax=ax)
                    st.pyplot(fig)
                elif pd.api.types.is_categorical_dtype(st.session_state["df"][column]) or st.session_state["df"][column].dtype == object:
                    st.write(st.session_state["df"][column].value_counts())
                    st.write("Bar Chart")
                    st.bar_chart(st.session_state["df"][column].value_counts())

        elif "recommend" in user_input.lower():
            # Extract features and target
            feature_columns = [col for col in st.session_state["df"].columns if col != 'crop']
            categorical_cols = st.session_state["df"].select_dtypes(include=['object']).columns.tolist()
            df_encoded = pd.get_dummies(st.session_state["df"], columns=categorical_cols, drop_first=True)
            X = df_encoded.drop('crop', axis=1, errors='ignore')
            y = st.session_state["df"]['crop']

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Model accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.session_state["model_accuracy"] = accuracy
            st.session_state["model"] = model
            st.session_state["label_encoder"] = label_encoder

            bot_response = f"Model trained successfully with an accuracy of {accuracy:.2f}! Enter feature values for prediction."
            append_chat(user_input, bot_response)

            # Input values for prediction
            input_data = {}
            for col in X.columns:
                input_data[col] = st.number_input(f"Enter {col} value", float(X[col].min()), float(X[col].max()))

            if st.button("Get Recommendation"):
                input_df = pd.DataFrame([input_data])
                prediction_encoded = model.predict(input_df)
                prediction = label_encoder.inverse_transform(prediction_encoded)
                bot_response = f"Recommended Crop: {prediction[0]}"
                append_chat("Prediction Input Submitted", bot_response)

        elif "model accuracy" in user_input.lower():
            if "model_accuracy" in st.session_state:
                accuracy = st.session_state["model_accuracy"]
                bot_response = f"The model accuracy is {accuracy:.2f}"
                append_chat(user_input, bot_response)
                
                # Displaying Model Accuracy and Confusion Matrix
                st.write("### Model Accuracy")
                st.write(f"Accuracy: {accuracy:.2f}")
                
                st.write("### Confusion Matrix")
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay.from_estimator(st.session_state["model"], X_test, y_test, display_labels=label_encoder.classes_, cmap="Blues", ax=ax)
                disp.plot(ax=ax)
                st.pyplot(fig)
            else:
                bot_response = "The model has not been trained yet. Please train the model first by asking for a crop recommendation."
                append_chat(user_input, bot_response)

        elif "top crops" in user_input.lower():
            top_crops = st.session_state["df"]['crop'].value_counts()
            bot_response = f"The top crops are:\n{top_crops}"
            append_chat(user_input, bot_response)
            st.write("### Crop Distribution")
            st.bar_chart(top_crops)

        elif "soil types" in user_input.lower():
            if 'soil type' in st.session_state["df"].columns:
                soil_types = st.session_state["df"]['soil type'].value_counts()
                bot_response = f"The most common soil types are:\n{soil_types}"
                append_chat(user_input, bot_response)
                st.write("### Soil Type Distribution")
                st.bar_chart(soil_types)
            else:
                bot_response = "The dataset does not contain a 'soil type' column."
                append_chat(user_input, bot_response)

        elif "correlation" in user_input.lower():
            numerical_cols = st.session_state["df"].select_dtypes(include=['number']).columns.tolist()
            if numerical_cols:
                correlation_matrix = st.session_state["df"][numerical_cols].corr()
                bot_response = "Here's the correlation heatmap of numerical features."
                append_chat(user_input, bot_response)
                st.write("### Correlation Heatmap")
                plt.figure(figsize=(12, 8))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
                plt.title('Correlation Matrix')
                st.pyplot(plt)
            else:
                bot_response = "There are no numerical features to compute correlations."
                append_chat(user_input, bot_response)

        else:
            bot_response = "I'm sorry, I didn't understand that. You can ask about data description, data summary, data analysis, model accuracy, top crops, soil types, correlations, or for a crop recommendation."
            append_chat(user_input, bot_response)

    # Display Chat History
    for chat in st.session_state["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
