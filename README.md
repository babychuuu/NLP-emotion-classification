# NLP-Emotion-Classification project

This repository presents a workflow for classifying text data related to mental health.  
It demonstrates two approaches:

1. **Traditional ML**  
   - Logistic Regression with CountVectorizer  
   - Data oversampling (RandomOverSampler)

2. **Deep Learning**  
   - Fine-tuning a DistilBERT Transformer model

## Overview

**Data Cleaning**: lowercasing, removing URLs, punctuation, numbers, and extra whitespace  
**Tokenization & Preprocessing**: NLTK-based tokenization, stopword removal, and stemming  

**Traditional Pipeline**:  
- Convert text to numeric features (`CountVectorizer`)  
- Address class imbalance via `RandomOverSampler`  
- Train and evaluate a Logistic Regression model

**Transformer Fine-Tuning**:  
- Use Hugging Face’s DistilBERT for sequence classification  
- Tokenize with `DistilBertTokenizerFast`, then train using the `Trainer` API  
- Evaluate and compare performance metrics (accuracy, precision, recall, F1-score)
