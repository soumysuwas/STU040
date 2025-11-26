# Reflection: CTF Challenge Methodology

## Approach and Strategy

My approach to this CTF challenge combined systematic data exploration, careful debugging, and machine learning analysis. The challenge required three distinct phases, each building upon the previous.

**Phase 1: Discovery** began with computing the SHA256 hash of my student ID and filtering the book dataset for specific rating criteria. Initially, I encountered a data structure mismatch—reviews linked to books via the `asin` field rather than `parent_asin`. Through methodical debugging and data exploration scripts, I discovered the hash "9E005560" in a review for "Alfie Cat in Trouble". Extracting FLAG1 required precise string manipulation to get the first 8 non-space characters ("AlfieCat") from the title.

**Phase 2: Identification** was straightforward once the hash was located. FLAG2 simply formatted the student hash, confirming the manipulated review.

**Phase 3: ML Analysis** presented the most technical complexity. I trained a Random Forest classifier to distinguish between suspicious reviews (short, superlative-heavy) and genuine reviews (detailed, domain-specific). Using TF-IDF vectorization captured meaningful text patterns. A critical insight was excluding the injected hash itself from potential authenticity indicators—the top words "forward", "incredible", and "alfie" genuinely reflected authentic review language about the book's narrative and characters. This demonstrates how feature importance can reveal authentic signals even in manipulated data.

The challenge reinforced the importance of thorough data exploration, careful debugging, and thoughtful feature engineering in data science investigations.
