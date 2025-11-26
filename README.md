# CTF Challenge: Find the Manipulated Book
**Student ID:** STU040

## Challenge Overview
This Capture the Flag challenge involved detecting a manipulated book review in a large dataset by acting as an AI detective. The goal was to identify the book, find a fake review, and explain authenticity using machine learning Analysis.

## Approach Summary

### FLAG1: Finding the Manipulated Book
1. **Hash Generation**: Computed SHA256 hash of student ID "STU040" and extracted first 8 characters: `9E005560`
2. **Book Filtering**: Searched for books with `rating_number = 1234` AND `average_rating = 5.0`
3. **Review Scanning**: Searched all reviews for the computed hash
4. **Discovery**: Found hash in a review for book "Alfie Cat in Trouble (Alfie A Friend for Life)" (ASIN: 0008172080)
5. **FLAG1 Extraction**: Took first 8 non-space characters ("AlfieCat"), computed SHA256

### FLAG2: Identifying the Fake Review
- Located the specific review containing the hash `9E005560`
- FLAG2 is simply the hash formatted as `FLAG2{9E005560}`

### FLAG3: Explaining Authenticity with ML
Used two complementary methods to identify authenticity indicators:

**Method A: Feature Importance (Random Forest)**
- Analyzed Gini impurity reduction in decision trees
- Top words: `forward`, `incredible`, `alfie`
- **FLAG3 (RF)**: `FLAG3{bba683e171}`

**Method B: SHAP Analysis (Game Theory)**
- Calculated marginal contribution of each word to "Genuine" prediction
- Top words: `alfie`, `best`, `book`
- **FLAG3 (SHAP)**: `FLAG3{192a60fa4f}`


## Results
```
FLAG1 = 513e114bd6035045c4570589ec333332f67b75c02480ed8b4dd45ecd4c10eef0
FLAG2 = FLAG2{9E005560}
FLAG3_RF = FLAG3{bba683e171}
FLAG3_SHAP = FLAG3{192a60fa4f}
```

## Repository Structure
```
STU040/
├── Files/
│   ├── books.csv          # Book dataset
│   └── reviews.csv        # Review dataset
├── solver.py              # Combined solution (RF + SHAP)
├── solver_shap.py         # Dedicated SHAP solver
├── flags.txt              # Final flags (both versions)
├── README.md              # This file
├── reflection.md          # Methodology reflection
└── SOLUTION_WALKTHROUGH.md # Detailed technical guide
```

## How to Run
```bash
# Run combined solver
python solver.py

# Run dedicated SHAP solver
python solver_shap.py
```

## Key Insights
1. **Data Matching Challenge**: Reviews use `asin` field while books use `parent_asin` for linking
2. **Hash Detection**: Case-insensitive search was critical for finding the manipulated review
3. **ML for Authenticity**: Feature importance from Random Forest effectively identified genuine review indicators
4. **Filter Requirements**: Excluding the planted hash from authenticity words was essential for FLAG3

## Author
Student ID: STU040
