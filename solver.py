"""
CTF Challenge Solver - Complete Solution
Student ID: STU040

This script solves all three flags of the CTF challenge:
- FLAG1: Find the manipulated book
- FLAG2: Identify the fake review hash
- FLAG3: Find authenticity indicators using ML

"""

import pandas as pd
import numpy as np
import hashlib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Constants
STUDENT_ID = "STU040"
STUDENT_ID_NUMERIC = "040"
BOOKS_FILE = r"E:\Antigravity\GrowthAXL\Files\books.csv"
REVIEWS_FILE = r"E:\Antigravity\GrowthAXL\Files\reviews.csv"

def compute_hash(text, length=8):
    """Compute SHA256 hash and return first N characters (uppercase)"""
    return hashlib.sha256(text.encode()).hexdigest()[:length].upper()

def find_flag1_and_flag2():
    """
    Step 1 & 2: Find the manipulated book and fake review
    """
    print("="*70)
    print("FINDING FLAG1 & FLAG2")
    print("="*70)
    
    # Compute student hash
    student_hash = compute_hash(STUDENT_ID, 8)
    print(f"\nStudent Hash: SHA256('{STUDENT_ID}')[:8] = {student_hash}")
    
    # Load data
    print(f"\nLoading datasets...")
    books_df = pd.read_csv(BOOKS_FILE)
    reviews_df = pd.read_csv(REVIEWS_FILE)
    reviews_df['text'] = reviews_df['text'].fillna('').astype(str)
    print(f"Loaded {len(books_df)} books and {len(reviews_df)} reviews")
    
    # Find review containing the hash
    print(f"\nSearching for hash '{student_hash}' in reviews...")
    mask = reviews_df['text'].str.contains(student_hash, case=False, regex=False)
    matching_reviews = reviews_df[mask]
    
    if len(matching_reviews) == 0:
        print("ERROR: No review found with hash!")
        return None, None, None
    
    review = matching_reviews.iloc[0]
    review_asin = review['asin']
    print(f"✓ Found review with ASIN: {review_asin}")
    
    # Find the book
    book_matches = books_df[books_df['parent_asin'] == review_asin]
    if len(book_matches) == 0:
        print(f"ERROR: No book found with parent_asin={review_asin}")
        return None, None, None
    
    book = book_matches.iloc[0]
    print(f"✓ Found book: {book['title']}")
    print(f"  Rating: {book['average_rating']}, Count: {book['rating_number']}")
    
    # Extract FLAG1
    book_title = book['title']
    non_space_chars = ''.join(book_title.split())[:8]
    flag1_hash = hashlib.sha256(non_space_chars.encode()).hexdigest()
    
    print(f"\nFLAG1 Computation:")
    print(f"  First 8 non-space chars: {non_space_chars}")
    print(f"  SHA256('{non_space_chars}'): {flag1_hash}")
    
    # FLAG2
    flag2 = f"FLAG2{{{student_hash}}}"
    
    print(f"\n✓ FLAG1 = {flag1_hash}")
    print(f"✓ FLAG2 = {flag2}")
    
    return flag1_hash, flag2, book['parent_asin'], student_hash

def find_flag3(book_asin, student_hash):
    """
    Step 3: Find authenticity indicators using ML
    """
    print("\n" + "="*70)
    print("FINDING FLAG3 - Machine Learning Analysis")
    print("="*70)
    
    # Load reviews
    reviews_df = pd.read_csv(REVIEWS_FILE)
    reviews_df['text'] = reviews_df['text'].fillna('').astype(str)
    
    # Get 5-star reviews for the book
    book_reviews = reviews_df[
        ((reviews_df['parent_asin'] == book_asin) | 
         (reviews_df['asin'] == book_asin)) &
        (reviews_df['rating'] == 5.0)
    ].copy()
    
    print(f"\nAnalyzing {len(book_reviews)} 5-star reviews")
    
    # Label suspicious vs genuine reviews
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
    
    print(f"Genuine: {book_reviews['is_genuine'].sum()}, "
          f"Suspicious: {(~book_reviews['is_genuine']).sum()}")

    
    # TF-IDF vectorization
    print("\nTraining classifier...")
    tfidf = TfidfVectorizer(
        max_features=150,
        stop_words='english',
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 1)
    )
    
    X_text = tfidf.fit_transform(book_reviews['text'])
    y = book_reviews['is_genuine'].astype(int)
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10
    )
    clf.fit(X_text.toarray(), y)
    
    # Get feature importances (exclude hash)
    feature_names = tfidf.get_feature_names_out()
    importances = clf.feature_importances_
    
    hash_lower = student_hash.lower()
    valid_features = [(i, name, imp) for i, (name, imp) in enumerate(zip(feature_names, importances))
                     if hash_lower not in name.lower()]
    valid_features.sort(key=lambda x: x[2], reverse=True)
    
    # Get top 3 authenticity words
    top_3_words = [word for _, word, _ in valid_features[:3]]
    
    print(f"\nTop 3 authenticity indicators: {', '.join(top_3_words)}")
    
    # Compute FLAG3
    flag3_input = ''.join(top_3_words) + STUDENT_ID_NUMERIC
    flag3_hash = hashlib.sha256(flag3_input.encode()).hexdigest()[:10]
    flag3 = f"FLAG3{{{flag3_hash}}}"
    
    print(f"\nFLAG3 Computation:")
    print(f"  Concatenated: {flag3_input}")
    print(f"  SHA256('{flag3_input}')[:10]: {flag3_hash}")
    print(f"\n✓ FLAG3 = {flag3}")
    
    return flag3

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("CTF CHALLENGE SOLVER")
    print(f"Student ID: {STUDENT_ID}")
    print("="*70)
    
    # Find FLAG1 and FLAG2
    flag1, flag2, book_asin, student_hash = find_flag1_and_flag2()
    
    if flag1 is None:
        print("\nFailed to find FLAG1 and FLAG2")
        return
    
    # Find FLAG3
    flag3 = find_flag3(book_asin, student_hash)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"FLAG1 = {flag1}")
    print(f"FLAG2 = {flag2}")
    print(f"FLAG3 = {flag3}")
    print("="*70)
    
    # Save to file
    with open(r"E:\Antigravity\GrowthAXL\flags.txt", "w") as f:
        f.write(f"FLAG1 = {flag1}\n")
        f.write(f"FLAG2 = {flag2}\n")
        f.write(f"FLAG3 = {flag3}\n")
    
    print("\nFlags saved to flags.txt")

if __name__ == "__main__":
    main()
