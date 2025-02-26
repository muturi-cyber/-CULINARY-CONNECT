import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define the preprocess_text function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into a string
    text = ' '.join(tokens)
    
    return text

# Load the TF-IDF Vectorizer and Logistic Regression Model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

# Define sentiment mapping
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Enter a text to analyze its sentiment (Positive, Neutral, Negative):")

# Input text
user_input = st.text_area("Input Text", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        
        # Transform the text using the TF-IDF Vectorizer
        text_tfidf = tfidf_vectorizer.transform([processed_text])
        
        # Predict the sentiment using the Logistic Regression model
        prediction = logistic_model.predict(text_tfidf)
        
        # Map the prediction to a sentiment label
        sentiment = sentiment_map.get(prediction[0], 'Unknown')
        
        # Display the result
        st.write(f"Sentiment: **{sentiment}**")
    else:
        st.write("Please enter some text to analyze.")

# CSV Upload Section
st.header("Upload a CSV File for Sentiment Analysis")
st.write("The CSV file should have a column named 'text' containing the text data to analyze.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Check if the 'text' column exists
    if 'text' in df.columns:
        st.write("File uploaded successfully! Here's a preview of the data:")
        st.write(df.head())

        # Fill NaN values with empty string and convert to string
        df['text'] = df['text'].fillna('').astype(str)

        # Analyze sentiment for each row, handling empty values safely
        def predict_sentiment(text):
            if text.strip():  # Only process if text is not empty
                processed_text = preprocess_text(text)
                text_tfidf = tfidf_vectorizer.transform([processed_text])
                prediction = logistic_model.predict(text_tfidf)[0]
                return sentiment_map.get(prediction, 'Unknown')
            else:
                return 'Unknown'

        # Apply function to the 'text' column
        df['sentiment'] = df['text'].apply(predict_sentiment)

        # Display the results
        st.write("Sentiment Analysis Results:")
        st.write(df)

        # Download the results as a CSV file
        st.download_button(
            label="Download Results as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='sentiment_analysis_results.csv',
            mime='text/csv'
        )
    else:
        st.error("The CSV file must contain a column named 'text'.")
