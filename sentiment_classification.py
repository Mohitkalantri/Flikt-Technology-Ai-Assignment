# -*- coding: utf-8 -*-
"""Sentiment Classification Model for Intelligent Customer Feedback Analysis System"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import os
import warnings
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

def map_sentiment(rating):
    """Map rating to sentiment"""
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

def create_svm_model(train_texts, train_labels, test_texts, test_labels):
    """Create and train a TF-IDF + SVM model"""
    print("Creating TF-IDF + SVM model...")
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('svm', SVC(kernel='linear', probability=True))
    ])
    
    # Train model
    print("Training TF-IDF + SVM model...")
    pipeline.fit(train_texts, train_labels)
    
    # Evaluate model
    predictions = pipeline.predict(test_texts)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))
    
    # Save model
    with open("sentiment_model_svm.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("✅ Simplified model saved as sentiment_model_svm.pkl")
    return pipeline

def main():
    print("=== Sentiment Classification Model ===")
    
    # Check if cleaned data exists
    if not os.path.exists("cleaned_feedback.csv"):
        print("❌ Cleaned feedback data not found. Please run data_preprocessing.py first.")
        return
    
    # Load cleaned data
    df = pd.read_csv("cleaned_feedback.csv")
    print(f"Loaded {len(df)} feedback records")
    
    # Map ratings to sentiments
    df['sentiment'] = df['rating'].apply(map_sentiment)
    print("Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Prepare training and testing data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['clean_text'].tolist(),
        df['sentiment'].tolist(),
        test_size=0.2,
        random_state=42
    )
    
    # Try to use DistilBERT first
    try:
        # Initialize tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        
        # Tokenize texts
        print("Tokenizing texts...")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
        
        # Create label mappings
        label2id = {'positive': 0, 'negative': 1, 'neutral': 2}
        id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
        
        train_labels_num = [label2id[label] for label in train_labels]
        test_labels_num = [label2id[label] for label in test_labels]
        
        # Create dataset class
        class FeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        # Create datasets
        train_dataset = FeedbackDataset(train_encodings, train_labels_num)
        test_dataset = FeedbackDataset(test_encodings, test_labels_num)
        
        # Initialize model
        print("Initializing model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=3,
            id2label=id2label,
            label2id=label2id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            report_to='none'
        )
        
        os.environ["WANDB_DISABLED"] = "true"
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer = tokenizer
        )
        
        # Train model
        print("Training model...")
        trainer.train()
        
        # Evaluate model
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        
        print("Accuracy:", accuracy_score(test_labels_num, preds))
        print(classification_report(test_labels_num, preds, target_names=list(label2id.keys())))
        
        # Save model
        model.save_pretrained("sentiment_model")
        tokenizer.save_pretrained("sentiment_model")
        print("✅ Model saved successfully!")
        
    except Exception as e:
        print(f"❌ Error with DistilBERT model: {e}")
        print("Falling back to TF-IDF + SVM model...")
        # Fallback to a simpler model using TF-IDF + SVM
        try:
            model = create_svm_model(train_texts, train_labels, test_texts, test_labels)
        except Exception as fallback_error:
            print(f"❌ Error creating fallback model: {fallback_error}")
            print("Please check your environment setup and dependencies.")

if __name__ == "__main__":
    main()