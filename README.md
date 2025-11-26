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
1. **Feature Engineering**: Created features to distinguish suspicious reviews (short + superlatives + hash) from genuine reviews (detailed + domain terms)
2. **Model Training**: Trained Random Forest classifier on 5-star reviews for the identified book
3. **TF-IDF Analysis**: Used TF-IDF vectorization to analyze text patterns
4. **Feature Importance**: Extracted top 3 words that indicate genuineness: "forward", "incredible", "alfie"
5. **FLAG3 Computation**: Concatenated words with numeric ID ("forwardincrediblealfie040") and computed SHA256

## Technical Stack
- **Python 3.9** with conda environment
- **pandas** & **numpy**: Data manipulation
- **scikit-learn**: Machine learning (RandomForestClassifier, TfidfVectorizer)
- **hashlib**: SHA256 hash computation

## Results
```
FLAG1 = 513e114bd6035045c4570589ec333332f67b75c02480ed8b4dd45ecd4c10eef0
FLAG2 = FLAG2{9E005560}
FLAG3 = FLAG3{bba683e171}
```

## Repository Structure
```
STU040/
├── Files/
│   ├── books.csv          # Book dataset
│   └── reviews.csv        # Review dataset
├── solver.py              # Complete solution code
├── flags.txt              # Final flags
├── README.md              # This file
└── reflection.md          # Methodology reflection
```

## How to Run
```bash
# Create and activate environment
conda create -n ctf_stu040 python=3.9 -y
conda activate ctf_stu040

# Install dependencies
pip install pandas numpy scikit-learn

# Run solver
python solver.py
```

## Key Insights
1. **Data Matching Challenge**: Reviews use `asin` field while books use `parent_asin` for linking
2. **Hash Detection**: Case-insensitive search was critical for finding the manipulated review
3. **ML for Authenticity**: Feature importance from Random Forest effectively identified genuine review indicators
4. **Filter Requirements**: Excluding the planted hash from authenticity words was essential for FLAG3

## Author
Student ID: STU040
