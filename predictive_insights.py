# -*- coding: utf-8 -*-
"""Predictive Insights Module for Intelligent Customer Feedback Analysis System"""

import pandas as pd
import numpy as np
from collections import Counter
import os

def main():
    print("=== Predictive Insight Generation ===")
    
    # Check if cleaned data exists
    if not os.path.exists("cleaned_feedback.csv"):
        print("❌ Cleaned feedback data not found. Please run data_preprocessing.py first.")
        return
    
    # Load cleaned data
    df = pd.read_csv("cleaned_feedback.csv")
    print(f"Loaded {len(df)} feedback records")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Map ratings to sentiments if not already done
    def map_sentiment(rating):
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['rating'].apply(map_sentiment)
    
    # Part 1: Identify recurring issues (most common words in negative feedback)
    print("\n--- Recurring Issues Analysis ---")
    negative_feedback = df[df['sentiment'] == 'negative']['clean_text']
    if len(negative_feedback) > 0:
        # Combine all negative feedback into one text
        all_negative_text = ' '.join(negative_feedback)
        # Split into words
        all_negative_words = all_negative_text.split()
        # Filter out short words and get most common
        filtered_words = [word for word in all_negative_words if len(word) > 3]
        word_freq = Counter(filtered_words)
        common_issues = word_freq.most_common(10)
        print("Top 10 recurring issues (common words in negative feedback):")
        for word, freq in common_issues:
            print(f"  - {word}: {freq} occurrences")
    else:
        print("No negative feedback found.")
    
    # Part 2: Sentiment trend analysis
    print("\n--- Sentiment Trend Analysis ---")
    # Group by month and sentiment
    df['month'] = df['date'].dt.to_period('M')
    sentiment_trend = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
    
    print("Sentiment trend by month:")
    print(sentiment_trend)
    
    # Predict next month's sentiment distribution (simple approach)
    print("\n--- Customer Satisfaction Prediction ---")
    if len(sentiment_trend) >= 2:
        # Calculate change from previous month to current month
        last_month = sentiment_trend.iloc[-1]
        prev_month = sentiment_trend.iloc[-2]
        change = last_month - prev_month
        
        # Predict next month
        next_month_prediction = last_month + change
        next_month_prediction = next_month_prediction.clip(lower=0)  # Ensure non-negative
        
        # Calculate satisfaction score (positive / total)
        total_current = last_month.sum()
        satisfaction_current = (last_month['positive'] / total_current) * 100 if total_current > 0 else 0
        
        total_predicted = next_month_prediction.sum()
        satisfaction_predicted = (next_month_prediction['positive'] / total_predicted) * 100 if total_predicted > 0 else 0
        
        print(f"Current month satisfaction rate: {satisfaction_current:.1f}%")
        print(f"Predicted next month satisfaction rate: {satisfaction_predicted:.1f}%")
        
        if satisfaction_predicted > satisfaction_current:
            print("📈 Customer satisfaction is predicted to improve next month!")
        elif satisfaction_predicted < satisfaction_current:
            print("📉 Customer satisfaction is predicted to decline next month!")
        else:
            print("➡️ Customer satisfaction is predicted to remain stable next month.")
    else:
        print("Insufficient data for trend prediction.")
    
    # Part 3: Source-based analysis
    print("\n--- Feedback Source Analysis ---")
    source_analysis = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
    print("Feedback distribution by source and sentiment:")
    print(source_analysis)
    
    # Calculate satisfaction rate by source
    source_analysis['total'] = source_analysis.sum(axis=1)
    source_analysis['satisfaction_rate'] = (source_analysis['positive'] / source_analysis['total']) * 100
    
    print("\nSatisfaction rate by source:")
    for source in source_analysis.index:
        rate = source_analysis.loc[source, 'satisfaction_rate']
        print(f"  {source}: {rate:.1f}%")
    
    print("\n✅ Predictive insights generated successfully!")

if __name__ == "__main__":
    main()