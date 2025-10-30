# -*- coding: utf-8 -*-
"""Text Summarization Module for Intelligent Customer Feedback Analysis System"""

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Download NLTK data files
nltk.download('punkt', quiet=True)

def extractive_summary(text, num_sentences=2):
    """Generate extractive summary using TF-IDF and cosine similarity"""
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate similarity scores
    similarity_matrix = cosine_similarity(tfidf_matrix)
    sentence_scores = similarity_matrix.sum(axis=1)
    
    # Get top sentences
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    top_sentences = [sentences[i] for i in sorted(top_indices)]
    
    return ' '.join(top_sentences)

def main():
    print("=== Text Summarization ===")
    
    # Check if cleaned data exists
    if not os.path.exists("cleaned_feedback.csv"):
        print("❌ Cleaned feedback data not found. Please run data_preprocessing.py first.")
        return
    
    # Load cleaned data
    df = pd.read_csv("cleaned_feedback.csv")
    print(f"Loaded {len(df)} feedback records")
    
    # Apply summarization to a subset
    df_subset = df.sample(min(100, len(df)), random_state=42)
    df_subset['short_summary'] = df_subset['clean_text'].apply(lambda x: extractive_summary(x, 1))
    df_subset['detailed_summary'] = df_subset['clean_text'].apply(lambda x: extractive_summary(x, 2))
    
    # Save summarized data
    df_subset.to_csv("summarized_feedback.csv", index=False)
    print("✅ Summarized feedback saved as summarized_feedback.csv")
    print("Rows in subset:", len(df_subset))
    
    # Display sample summaries
    print("\nSample summaries:")
    sample_feedbacks = df_subset.sample(min(3, len(df_subset)), random_state=42)
    for i, row in sample_feedbacks.iterrows():
        print("="*80)
        print(f"🗣️ Original Feedback:\n{row['clean_text']}\n")
        print(f"✂️ Short Summary: {row['short_summary']}\n")
        print(f"📝 Detailed Summary: {row['detailed_summary']}\n")

if __name__ == "__main__":
    main()