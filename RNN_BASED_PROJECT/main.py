import re
import requests
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import os

# Settings
MAX_FEATURES = 10000
MAXLEN = 500

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36"
}

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current working directory:", os.getcwd())

# Load model
try:
    model_path = 'simple_rnn_imdb.h5'
    print(f"Attempting to load model from: {os.path.abspath(model_path)}")
    model = load_model(model_path)
    st.success("Model loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Model file not found at {os.path.abspath(model_path)}. Check directory: {os.getcwd()}")
    raise
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise

def fetch_reviews(url):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        review_texts = []
        review_containers = soup.find_all("div", class_="review-container")
        for container in review_containers:
            text_div = container.find("div", class_="text")
            if text_div:
                review_text = text_div.get_text(separator=" ", strip=True)
                cleaned_text = re.sub(r"\s+", " ", review_text).strip()
                if len(cleaned_text) > 50:
                    review_texts.append(cleaned_text)
        return review_texts if review_texts else None
    except Exception as e:
        st.error(f"Error fetching reviews: {e}")
        return None

def search_movie_review_page(movie_name):
    query = movie_name.replace(" ", "+")
    search_url = f"https://www.imdb.com/find/?q={query}&s=tt"
    response = requests.get(search_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    result = soup.find("a", class_="ipc-metadata-list-summary-item__t")
    if not result:
        raise Exception("Movie not found.")
    movie_link = "https://www.imdb.com" + result["href"]
    reviews_url = movie_link.rstrip("/") + "/reviews"
    return reviews_url

def preprocess_text(text, word_index, maxlen=MAXLEN, max_features=MAX_FEATURES):
    words = text.lower().split()
    encoded_review = [1]  # Start token
    for word in words:
        index = word_index.get(word, 2) + 3
        if index >= max_features:
            index = 2
        encoded_review.append(index)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen, padding='post', truncating='post')
    return padded_review

def main():
    st.title("Movie Review Sentiment Analyzer")
    st.write("Enter a movie name to analyze the top 5 IMDb reviews.")

    movie_name = st.text_input("Enter a movie name (e.g., The Shawshank Redemption):", "")
    
    if st.button("Analyze Reviews"):
        if not movie_name:
            st.error("Please enter a movie name.")
            return

        with st.spinner("Fetching movie reviews..."):
            try:
                reviews_url = search_movie_review_page(movie_name)
                st.write(f"Fetched reviews page URL: {reviews_url}")
            except Exception as e:
                st.error(f"Error fetching movie: {e}")
                return

            reviews = fetch_reviews(reviews_url)
            if not reviews:
                st.warning("No reviews found. Using mock reviews instead.")
                reviews = [
                    "This movie is a masterpiece, incredible performances by all.",
                    "The plot was slow, didn’t enjoy it as much as expected.",
                    "One of the best films ever, a powerful story of hope.",
                    "Felt overhyped, predictable and not that great.",
                    "Beautifully crafted, deeply moving film."
                ]
                st.write(f"Using {len(reviews)} mock reviews.")

        word_index = imdb.get_word_index()

        st.subheader(f"Top 5 Reviews for '{movie_name}':")
        sentiments = []
        scores = []
        for i, review in enumerate(reviews[:5], 1):
            preprocessed_input = preprocess_text(review, word_index)
            preprocessed_input = preprocessed_input.astype('int32')
            try:
                prediction = model.predict(preprocessed_input, verbose=0)
                score = prediction[0][0]
                sentiment = "Positive" if score > 0.5 else "Negative"
                sentiments.append(sentiment)
                scores.append(score)
                st.write(f"**Review {i}:** {review[:100]}...")
                st.write(f"Sentiment: {sentiment}, Score: {score:.4f}")
                st.write("---")
            except Exception as e:
                st.error(f"Error predicting sentiment for review {i}: {e}")

        if scores:
            avg_score = np.mean(scores)
            avg_sentiment = "Positive" if avg_score > 0.5 else "Negative"
            st.subheader("Summary")
            st.write(f"Average Sentiment for '{movie_name}' (based on top 5 reviews): **{avg_sentiment}**")
            st.write(f"Average Prediction Score: **{avg_score:.4f}**")
        else:
            st.error("No valid reviews to analyze.")

if __name__ == "__main__":
    main()
