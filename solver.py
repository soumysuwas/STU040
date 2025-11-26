"""
CTF Challenge Solver - Complete Solution (Dual Method)
Student ID: STU040

This script solves all three flags and computes FLAG3 using TWO methods:
1. Random Forest Feature Importance (Original)
2. SHAP Analysis (Advanced)
"""

import pandas as pd
import numpy as np
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import shap
import warnings

warnings.filterwarnings('ignore')

# Constants
STUDENT_ID = "STU040"
STUDENT_ID_NUMERIC = "040"
BOOKS_FILE = r"E:\Antigravity\GrowthAXL\Files\books.csv"
REVIEWS_FILE = r"E:\Antigravity\GrowthAXL\Files\reviews.csv"

def compute_hash(text, length=8):
    return hashlib.sha256(text.encode()).hexdigest()[:length].upper()

def find_flags():
    print("="*70)
    print("CTF CHALLENGE SOLVER - DUAL METHOD")
    print("="*70)
    
    # --- FLAG 1 & 2 ---
    student_hash = compute_hash(STUDENT_ID, 8)
    print(f"\n[1] Finding Book & Fake Review (Hash: {student_hash})")
    
    books_df = pd.read_csv(BOOKS_FILE)
    reviews_df = pd.read_csv(REVIEWS_FILE)
    reviews_df['text'] = reviews_df['text'].fillna('').astype(str)
    
    mask = reviews_df['text'].str.contains(student_hash, case=False, regex=False)
    matching_reviews = reviews_df[mask]
    
    if len(matching_reviews) == 0:
        print("Error: Review not found")
        return
        
    review = matching_reviews.iloc[0]
    book = books_df[books_df['parent_asin'] == review['asin']].iloc[0]
    
    book_title = book['title']
    non_space_chars = ''.join(book_title.split())[:8]
    flag1 = hashlib.sha256(non_space_chars.encode()).hexdigest()
    flag2 = f"FLAG2{{{student_hash}}}"
    
    print(f"    Book: {book_title}")
    print(f"    FLAG1: {flag1}")
    print(f"    FLAG2: {flag2}")
    
    # --- FLAG 3 PREP ---
    print(f"\n[2] Preparing Data for FLAG3")
    book_reviews = reviews_df[
        ((reviews_df['parent_asin'] == book['parent_asin']) | 
         (reviews_df['asin'] == book['parent_asin'])) &
        (reviews_df['rating'] == 5.0)
    ].copy()
    
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
    
    tfidf = TfidfVectorizer(max_features=150, stop_words='english', min_df=1, max_df=0.9)
    X_text = tfidf.fit_transform(book_reviews['text'])
    X_dense = X_text.toarray().astype(np.float64)
    y = book_reviews['is_genuine'].astype(int)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_dense, y)
    feature_names = tfidf.get_feature_names_out()
    
    # --- METHOD A: FEATURE IMPORTANCE ---
    print(f"\n[3] Method A: Random Forest Feature Importance")
    importances = clf.feature_importances_
    hash_lower = student_hash.lower()
    
    rf_features = [(name, imp) for name, imp in zip(feature_names, importances) 
                  if hash_lower not in name.lower()]
    rf_features.sort(key=lambda x: x[1], reverse=True)
    
    top_rf = [w for w, s in rf_features[:3]]
    flag3_rf_input = ''.join(top_rf) + STUDENT_ID_NUMERIC
    flag3_rf = f"FLAG3{{{hashlib.sha256(flag3_rf_input.encode()).hexdigest()[:10]}}}"
    
    print(f"    Top Words: {', '.join(top_rf)}")
    print(f"    FLAG3 (RF): {flag3_rf}")
    
    # --- METHOD B: SHAP ANALYSIS ---
    print(f"\n[4] Method B: SHAP Analysis")
    try:
        if len(X_dense) < 10:
            background = X_dense
        else:
            background = shap.kmeans(X_dense, 10)
            
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_dense, check_additivity=False)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values
            
        mean_shap = shap_vals.mean(axis=0)
        
        shap_features = []
        for name, score in zip(feature_names, mean_shap):
            if hash_lower not in name.lower():
                shap_features.append((name, float(score)))
        
        shap_features.sort(key=lambda x: x[1], reverse=True)
        
        top_shap = [w for w, s in shap_features[:3]]
        flag3_shap_input = ''.join(top_shap) + STUDENT_ID_NUMERIC
        flag3_shap = f"FLAG3{{{hashlib.sha256(flag3_shap_input.encode()).hexdigest()[:10]}}}"
        
        print(f"    Top Words: {', '.join(top_shap)}")
        print(f"    FLAG3 (SHAP): {flag3_shap}")
        
    except Exception as e:
        print(f"    SHAP Analysis failed: {e}")
        flag3_shap = "ERROR"

    # --- SAVE RESULTS ---
    print(f"\n[5] Saving Results")
    with open(r"E:\Antigravity\GrowthAXL\flags.txt", "w") as f:
        f.write(f"FLAG1 = {flag1}\n")
        f.write(f"FLAG2 = {flag2}\n")
        f.write(f"FLAG3_RF = {flag3_rf}\n")
        f.write(f"FLAG3_SHAP = {flag3_shap}\n")
    print("    Saved to flags.txt")

if __name__ == "__main__":
    find_flags()
