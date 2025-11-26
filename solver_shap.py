"""
CTF Challenge Solver - SHAP Method
Student ID: STU040

This script solves the challenge using SHAP (SHapley Additive exPlanations)
for the authenticity analysis (FLAG3).
"""

import pandas as pd
import numpy as np
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import shap
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
STUDENT_ID = "STU040"
STUDENT_ID_NUMERIC = "040"
BOOKS_FILE = r"E:\Antigravity\GrowthAXL\Files\books.csv"
REVIEWS_FILE = r"E:\Antigravity\GrowthAXL\Files\reviews.csv"

def compute_hash(text, length=8):
    return hashlib.sha256(text.encode()).hexdigest()[:length].upper()

def solve():
    print("="*70)
    print("CTF SOLVER - SHAP METHOD")
    print("="*70)

    # --- FLAG 1 & 2 (Simplified for this script) ---
    student_hash = compute_hash(STUDENT_ID, 8)
    print(f"Student Hash: {student_hash}")
    
    # We know the book from previous exploration
    book_asin = '0008172080' 
    print(f"Target Book ASIN: {book_asin}")

    # --- FLAG 3: SHAP Analysis ---
    print("\n" + "-"*30)
    print("Computing FLAG3 using SHAP")
    print("-" * 30)

    # Load reviews
    print("Loading reviews...")
    reviews_df = pd.read_csv(REVIEWS_FILE)
    reviews_df['text'] = reviews_df['text'].fillna('').astype(str)

    # Filter 5-star reviews
    book_reviews = reviews_df[
        ((reviews_df['parent_asin'] == book_asin) | 
         (reviews_df['asin'] == book_asin)) &
        (reviews_df['rating'] == 5.0)
    ].copy()
    print(f"Found {len(book_reviews)} 5-star reviews")

    # Labeling logic
    def is_suspicious(text, hash_str):
        text_lower = str(text).lower()
        words = text_lower.split()
        superlatives = ['best', 'amazing', 'incredible', 'perfect', 'awesome']
        has_superlatives = sum(1 for sup in superlatives if sup in text_lower)
        is_short = len(words) < 15
        has_hash = hash_str.lower() in text_lower
        return ((is_short and has_superlatives >= 2) or has_hash)

    book_reviews['is_genuine'] = book_reviews['text'].apply(
        lambda x: not is_suspicious(x, student_hash)
    )

    # TF-IDF
    print("Vectorizing text...")
    tfidf = TfidfVectorizer(max_features=150, stop_words='english', min_df=1, max_df=0.9)
    X_text = tfidf.fit_transform(book_reviews['text'])
    X_dense = X_text.toarray().astype(np.float64)
    y = book_reviews['is_genuine'].astype(int)

    # Train Model
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    clf.fit(X_dense, y)

    # SHAP Analysis
    print("Calculating SHAP values...")
    try:
        # Handle small sample size
        if len(X_dense) < 10:
            background = X_dense
        else:
            background = shap.kmeans(X_dense, 10)
        
        explainer = shap.TreeExplainer(clf)
        
        # Calculate SHAP for genuine class
        # Use all data if small, else sample
        if len(X_dense) > 20:
            genuine_indices = np.where(y == 1)[0]
            if len(genuine_indices) > 0:
                sample_indices = np.random.choice(genuine_indices, min(20, len(genuine_indices)), replace=False)
                X_sample = X_dense[sample_indices]
            else:
                X_sample = X_dense
        else:
            X_sample = X_dense

        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        # Handle output format
        if isinstance(shap_values, list):
            shap_values_genuine = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_genuine = shap_values[:, :, 1]
        else:
            shap_values_genuine = shap_values

        # Mean SHAP (signed)
        mean_shap = shap_values_genuine.mean(axis=0)
        feature_names = tfidf.get_feature_names_out()

        # Filter and Sort
        hash_lower = student_hash.lower()
        valid_features = []
        for name, score in zip(feature_names, mean_shap):
            if hash_lower not in name.lower():
                valid_features.append((name, float(score)))
        
        valid_features.sort(key=lambda x: x[1], reverse=True)
        
        top_3_words = [w for w, s in valid_features[:3]]
        print(f"\nTop 3 SHAP words: {', '.join(top_3_words)}")

        # Compute FLAG3
        flag3_input = ''.join(top_3_words) + STUDENT_ID_NUMERIC
        flag3_hash = hashlib.sha256(flag3_input.encode()).hexdigest()[:10]
        flag3 = f"FLAG3{{{flag3_hash}}}"
        
        print(f"FLAG3 (SHAP) = {flag3}")
        return flag3

    except Exception as e:
        print(f"SHAP Analysis failed: {e}")
        return None

if __name__ == "__main__":
    solve()
