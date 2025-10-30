# -*- coding: utf-8 -*-
"""Data Preprocessing Module for Intelligent Customer Feedback Analysis System"""

import pandas as pd
import numpy as np
import random
import nltk
from faker import Faker
import datetime
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data files
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

fake = Faker()

def generate_feedback(num_records=1200):
    """Generate synthetic customer feedback data"""
    sources = ['Email', 'Chat', 'Twitter', 'Facebook', 'Survey']
    positive_comments = [
        "Great product quality!", "Fast delivery, very happy!", "Excellent customer support.",
        "Loved the packaging!", "Highly satisfied with the purchase."
    ]
    negative_comments = [
        "Product arrived damaged.", "Customer support was unhelpful.",
        "Delivery was delayed.", "Refund process took too long.",
        "Received the wrong item."
    ]
    neutral_comments = [
        "It's okay, nothing special.", "Product is average.",
        "Not bad, but could be better.", "Fair experience overall."
    ]

    data = []
    for i in range(num_records):
        source = random.choice(sources)
        date = datetime.date.today() - datetime.timedelta(days=random.randint(0, 120))
        sentiment = random.choice(["positive", "negative", "neutral"])
        if sentiment == "positive":
            text = random.choice(positive_comments) + " " + fake.sentence()
            rating = random.choice([4, 5])
        elif sentiment == "negative":
            text = random.choice(negative_comments) + " " + fake.sentence()
            rating = random.choice([1, 2])
        else:
            text = random.choice(neutral_comments) + " " + fake.sentence()
            rating = 3

        data.append([i+1, source, date, text, rating])

    df = pd.DataFrame(data, columns=['feedback_id', 'source', 'date', 'text', 'rating'])
    return df

def clean_text(text):
    """Clean text by removing URLs, special characters, and extra spaces"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text):
    """Preprocess text with tokenization, lemmatization, and stopword removal"""
    STOPWORDS = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Tokenize and clean text
    tokens = word_tokenize(clean_text(text))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def main():
    print("=== Data Preprocessing for Customer Feedback Analysis ===")
    
    # Generate feedback data
    df = generate_feedback()
    print(f"Generated {len(df)} feedback records")
    
    # Remove duplicates and missing values
    df.drop_duplicates(subset='text', inplace=True)
    df.dropna(subset=['text'], inplace=True)
    print(f"After removing duplicates and NaN: {len(df)} records")
    
    # Apply cleaning
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Save cleaned dataset
    df.to_csv("cleaned_feedback.csv", index=False)
    print("✅ Cleaned dataset saved as cleaned_feedback.csv")
    print("Rows:", len(df))
    
    # Display sample of cleaned data
    print("\nSample of cleaned data:")
    print(df[['feedback_id', 'source', 'text', 'clean_text', 'rating']].head())

if __name__ == "__main__":
    main()