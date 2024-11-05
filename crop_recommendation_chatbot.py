import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Interactive Crop Recommendation Chatbot")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Step 1: Upload Your Dataset
st.write("### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your agricultural CSV file (with features like soil type, temperature, crop)", type="csv")

# Check if a file is uploaded and can be read
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Verify the dataset has content and columns
        if df.empty:
            st.write("Uploaded file is empty. Please upload a valid CSV file.")
        else:
            # Save the DataFrame in session state
            st.session_state["df"] = df
            st.write("### Dataset Preview")
            st.write(df.head())

            # Data Analysis Section
            st.write("### Data Analysis")
            st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            st.write("#### Summary Statistics")
            st.write(df.describe(include='all'))

            # Analyzing Categorical Features
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.write("#### Categorical Features Analysis")
                for col in categorical_cols:
                    st.write(f"**{col}**")
                    st.write(df[col].value_counts())
                    plt.figure(figsize=(10, 5))
                    sns.countplot(data=df, x=col)
                    plt.title(f'Distribution of {col}')
                    plt.xticks(rotation=45)
                    st.pyplot(plt)

            # Analyzing Numerical Features
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numerical_cols:
                st.write("#### Numerical Features Analysis")
                for col in numerical_cols:
                    st.write(f"**{col}**")
                    st.write(df[col].describe())
                    plt.figure(figsize=(10, 5))
                    sns.histplot(df[col], bins=20, kde=True)
                    plt.title(f'Distribution of {col}')
                    st.pyplot(plt)

            # Check correlation between features if numerical columns exist
            if numerical_cols:
                st.write("#### Correlation Heatmap")
                plt.figure(figsize=(12, 8))
                correlation_matrix = df[numerical_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
                plt.title('Correlation Matrix')
                st.pyplot(plt)

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
            bot_response = "Hello! Your dataset is loaded. You can ask me to recommend a crop or get data insights."
            append_chat(user_input, bot_response)

        elif 'crop' not in st.session_state["df"].columns:
            bot_response = "Your dataset must include a 'crop' column as the target variable."
            append_chat(user_input, bot_response)

        elif "recommend" in user_input.lower() or "crop" in user_input.lower():
            feature_columns = [col for col in st.session_state["df"].columns if col != 'crop']
            
            # Encode categorical features
            df_encoded = pd.get_dummies(st.session_state["df"], columns=categorical_cols, drop_first=True)

            # Separate features and target variable
            if 'crop' in df_encoded.columns:
                X = df_encoded.drop('crop', axis=1)
                y = df_encoded['crop']
            else:
                bot_response = "The target variable 'crop' is not found after encoding."
                append_chat(user_input, bot_response)
                st.stop()

            # Train a model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            joblib.dump(model, 'user_crop_model.pkl')

            # Prompt for input data
            bot_response = "Model trained! Now, enter data for a crop recommendation."
            append_chat(user_input, bot_response)

            # Prompt user for each feature
            input_data = {}
            for col in X.columns:
                input_data[col] = st.number_input(f"Enter {col} value", float(X[col].min()), float(X[col].max()))

            # Get Prediction
            if st.button("Get Recommendation"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                bot_response = f"Recommended Crop: {prediction[0]}"
                append_chat(user_input, bot_response)

        elif "insight" in user_input.lower():
            bot_response = "You can ask for crop distribution, summary statistics, or correlation insights!"
            append_chat(user_input, bot_response)

        else:
            bot_response = "Please ask for a crop recommendation, insights, or say 'hello' to start."
            append_chat(user_input, bot_response)

    # Display Chat History
    for chat in st.session_state["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Interactive Crop Recommendation Chatbot: AGRINEXUS")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Step 1: Upload Your Dataset
st.write("### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your agricultural CSV file (with features like soil type, temperature, crop)", type="csv")

# Check if a file is uploaded and can be read
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Verify the dataset has content and columns
        if df.empty:
            st.write("Uploaded file is empty. Please upload a valid CSV file.")
        else:
            # Save the DataFrame in session state
            st.session_state["df"] = df
            st.write("### Dataset Preview")
            st.write(df.head())

            # Data Analysis Section
            st.write("### Data Analysis")
            st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            st.write("#### Summary Statistics")
            st.write(df.describe(include='all'))

            # Analyzing Categorical Features
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.write("#### Categorical Features Analysis")
                for col in categorical_cols:
                    st.write(f"**{col}**")
                    st.write(df[col].value_counts())
                    plt.figure(figsize=(10, 5))
                    sns.countplot(data=df, x=col)
                    plt.title(f'Distribution of {col}')
                    plt.xticks(rotation=45)
                    st.pyplot(plt)

            # Analyzing Numerical Features
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numerical_cols:
                st.write("#### Numerical Features Analysis")
                for col in numerical_cols:
                    st.write(f"**{col}**")
                    st.write(df[col].describe())
                    plt.figure(figsize=(10, 5))
                    sns.histplot(df[col], bins=20, kde=True)
                    plt.title(f'Distribution of {col}')
                    st.pyplot(plt)

            # Check correlation between features if numerical columns exist
            if numerical_cols:
                st.write("#### Correlation Heatmap")
                plt.figure(figsize=(12, 8))
                correlation_matrix = df[numerical_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
                plt.title('Correlation Matrix')
                st.pyplot(plt)

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
            bot_response = "Hello! Your dataset is loaded. You can ask me to recommend a crop or get data insights."
            append_chat(user_input, bot_response)

        elif 'crop' not in st.session_state["df"].columns:
            bot_response = "Your dataset must include a 'crop' column as the target variable."
            append_chat(user_input, bot_response)

        elif "recommend" in user_input.lower() or "crop" in user_input.lower():
            feature_columns = [col for col in st.session_state["df"].columns if col != 'crop']
            
            # Encode categorical features
            df_encoded = pd.get_dummies(st.session_state["df"], columns=categorical_cols, drop_first=True)
            X = df_encoded.drop('crop', axis=1)
            y = df_encoded['crop']

            # Train a model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            joblib.dump(model, 'user_crop_model.pkl')

            # Prompt for input data
            bot_response = "Model trained! Now, enter data for a crop recommendation."
            append_chat(user_input, bot_response)

            # Prompt user for each feature
            input_data = {}
            for col in X.columns:
                input_data[col] = st.number_input(f"Enter {col} value", float(X[col].min()), float(X[col].max()))

            # Get Prediction
            if st.button("Get Recommendation"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                bot_response = f"Recommended Crop: {prediction[0]}"
                append_chat(user_input, bot_response)

        elif "insight" in user_input.lower():
            bot_response = "You can ask for crop distribution, summary statistics, or correlation insights!"
            append_chat(user_input, bot_response)

        else:
            bot_response = "Please ask for a crop recommendation, insights, or say 'hello' to start."
            append_chat(user_input, bot_response)

    # Display Chat History
    for chat in st.session_state["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
