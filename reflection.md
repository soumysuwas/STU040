# Reflection: CTF Challenge Methodology

## Approach and Strategy

My approach to this CTF challenge combined systematic data exploration, careful debugging, and machine learning analysis. The challenge required three distinct phases, each building upon the previous.

**Phase 1: Discovery** began with computing the SHA256 hash of my student ID and filtering the book dataset for specific rating criteria. Initially, I encountered a data structure mismatch—reviews linked to books via the `asin` field rather than `parent_asin`. Through methodical debugging and data exploration scripts, I discovered the hash "9E005560" in a review for "Alfie Cat in Trouble". Extracting FLAG1 required precise string manipulation to get the first 8 non-space characters ("AlfieCat") from the title.

**Phase 2: Identification** was straightforward once the hash was located. FLAG2 simply formatted the student hash, confirming the manipulated review.

**Phase 3: ML Analysis** presented the most technical complexity. I employed two distinct approaches to ensure robustness:

1. **Random Forest Feature Importance**: Identified "forward", "incredible", "alfie".
2. **SHAP Analysis**: Identified "alfie", "best", "book".

Both methods agreed on "alfie" as a core authenticity signal, validating the model's learning. SHAP provided a more nuanced view of marginal contributions, while Feature Importance highlighted tree-split utility. I included both results to demonstrate a comprehensive understanding of model interpretability. A critical insight in both cases was excluding the injected hash itself from potential authenticity indicators.


The challenge reinforced the importance of thorough data exploration, careful debugging, and thoughtful feature engineering in data science investigations.
