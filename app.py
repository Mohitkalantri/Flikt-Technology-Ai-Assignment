# -*- coding: utf-8 -*-
"""Flask Web App for Intelligent Customer Feedback Analysis System"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import os
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import datetime
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Download NLTK data files
nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Global variables to store model and data
model = None
df = None

def create_fallback_model():
    """Create a simple fallback model if the trained model is not available"""
    global model
    # Create a simple rule-based model as fallback
    class SimpleModel:
        def predict(self, texts):
            predictions = []
            for text in texts:
                text = text.lower()
                if any(word in text for word in ['great', 'good', 'excellent', 'love', 'happy', 'satisfied']):
                    predictions.append('positive')
                elif any(word in text for word in ['bad', 'terrible', 'awful', 'hate', 'angry', 'disappointed']):
                    predictions.append('negative')
                else:
                    predictions.append('neutral')
            return predictions
        
        def predict_proba(self, texts):
            # Return uniform probabilities as fallback
            probs = []
            for _ in texts:
                probs.append([0.33, 0.33, 0.34])  # Equal probabilities
            return np.array(probs)
    
    model = SimpleModel()
    print("✅ Fallback model created!")

def load_model():
    """Load the trained sentiment model"""
    global model
    try:
        with open("sentiment_model_svm.pkl", "rb") as f:
            model = pickle.load(f)
        print("✅ Sentiment model loaded successfully!")
    except FileNotFoundError:
        print("❌ Sentiment model not found. Creating fallback model.")
        create_fallback_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        create_fallback_model()

def load_data():
    """Load the feedback data"""
    global df
    try:
        df = pd.read_csv("cleaned_feedback.csv")
        print(f"✅ Loaded {len(df)} feedback records!")
    except FileNotFoundError:
        print("❌ Cleaned feedback data not found. Run data_preprocessing.py first.")
        df = None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        df = None

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

def map_sentiment(rating):
    """Map rating to sentiment"""
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a single piece of feedback"""
    feedback_text = request.form.get('feedback')
    
    if not feedback_text or not model:
        return jsonify({'error': 'Missing feedback text or model not loaded'}), 400
    
    # Predict sentiment
    prediction = model.predict([feedback_text])[0]
    
    # Get prediction probability
    try:
        probabilities = model.predict_proba([feedback_text])[0]
        confidence = max(probabilities)
    except:
        confidence = 1.0  # If probability not available
    
    # Generate summary
    summary = extractive_summary(feedback_text, 1)
    
    return jsonify({
        'sentiment': prediction,
        'confidence': float(confidence),
        'summary': summary
    })

@app.route('/upload', methods=['POST'])
def upload():
    """Upload and analyze a CSV file of feedback"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file.filename is not None and is a string before using endswith
    if file and file.filename and hasattr(file.filename, 'endswith') and file.filename.endswith('.csv'):
        # Save file temporarily
        filepath = 'temp_upload.csv'
        file.save(filepath)
        
        try:
            # Read CSV
            uploaded_df = pd.read_csv(filepath)
            
            # Check if required columns exist
            if 'text' not in uploaded_df.columns and 'feedback' not in uploaded_df.columns:
                return jsonify({'error': 'CSV must contain a "text" or "feedback" column'}), 400
            
            # Use appropriate column
            text_column = 'text' if 'text' in uploaded_df.columns else 'feedback'
            
            # Analyze sentiments
            if model:
                uploaded_df['sentiment'] = model.predict(uploaded_df[text_column])
                uploaded_df['summary'] = uploaded_df[text_column].apply(lambda x: extractive_summary(str(x), 1))
            else:
                # Fallback to rating-based sentiment if model not available
                if 'rating' in uploaded_df.columns:
                    uploaded_df['sentiment'] = uploaded_df['rating'].apply(map_sentiment)
                else:
                    uploaded_df['sentiment'] = 'neutral'  # Default if no rating
                uploaded_df['summary'] = uploaded_df[text_column].apply(lambda x: extractive_summary(str(x), 1))
            
            # Convert to JSON for response
            result = uploaded_df.to_dict(orient='records')
            
            # Clean up temp file
            os.remove(filepath)
            
            return jsonify(result)
        except Exception as e:
            # Clean up temp file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

@app.route('/insights')
def insights():
    """Get analytical insights"""
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 400
    
    # Prepare data for analysis
    analysis_df = df.copy()
    
    # Convert date column to datetime
    analysis_df['date'] = pd.to_datetime(analysis_df['date'])
    
    # Map ratings to sentiments if not already done
    if 'sentiment' not in analysis_df.columns:
        analysis_df['sentiment'] = analysis_df['rating'].apply(map_sentiment)
    
    # Sentiment distribution
    sentiment_dist = analysis_df['sentiment'].value_counts().to_dict()
    
    # Source distribution
    source_dist = analysis_df['source'].value_counts().to_dict()
    
    # Sentiment by source
    sentiment_by_source = analysis_df.groupby(['source', 'sentiment']).size().unstack(fill_value=0).to_dict()
    
    # Trend analysis (by month)
    analysis_df['month'] = analysis_df['date'].dt.to_period('M').astype(str)
    trend_data = analysis_df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
    trend_dict = trend_data.reset_index().to_dict(orient='records')
    
    # Recurring issues (from negative feedback)
    negative_feedback = analysis_df[analysis_df['sentiment'] == 'negative']['clean_text']
    if len(negative_feedback) > 0:
        all_negative_text = ' '.join(negative_feedback)
        all_negative_words = all_negative_text.split()
        filtered_words = [word for word in all_negative_words if len(word) > 3]
        word_freq = Counter(filtered_words)
        common_issues = dict(word_freq.most_common(10))
    else:
        common_issues = {}
    
    return jsonify({
        'sentiment_distribution': sentiment_dist,
        'source_distribution': source_dist,
        'sentiment_by_source': sentiment_by_source,
        'trend_data': trend_dict,
        'common_issues': common_issues
    })

if __name__ == '__main__':
    # Load model and data on startup
    load_model()
    load_data()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)