# NLP-Emotion-Classification

This repository presents a workflow for classifying text data related to mental health using both **traditional machine learning** and **deep learning** approaches. It leverages **5-fold cross-validation** for more robust model evaluation.

We compare:

1. **Traditional ML**  
   - Logistic Regression with CountVectorizer  
   - Data oversampling (RandomOverSampler)  
   - Evaluated via 5-fold CV

2. **Deep Learning**  
   - Fine-tuning a DistilBERT Transformer model  
   - Tokenization with `DistilBertTokenizerFast`  
   - Trained via the Hugging Face `Trainer` API  
   - Also evaluated with 5-fold CV

Our results (as shown in the attached confusion matrices) demonstrate **improved performance** after fine-tuning DistilBERT (**blue heatmap**) compared to Logistic Regression (**green heatmap**).

---

## Overview

### Data Cleaning
- Lowercasing, removing URLs, punctuation, numbers, and extra whitespace.

### Tokenization & Preprocessing
- NLTK-based tokenization, stopword removal, and stemming.

### Traditional Pipeline (Logistic Regression)
1. **Feature Extraction**: `CountVectorizer` transforms text into numeric features.  
2. **Class Imbalance**: `RandomOverSampler` addresses minority class imbalance.  
3. **Training & Evaluation**: Logistic Regression is trained and evaluated using **5-fold cross-validation**.

### Transformer Fine-Tuning (DistilBERT)
1. **Model**: `DistilBertForSequenceClassification` for text classification.  
2. **Tokenizer**: `DistilBertTokenizerFast` (max sequence length = 128).  
3. **Training**: Hugging Face `Trainer` API.  
4. **5-Fold CV**: Same cross-validation splits for consistent comparison.  
5. **Outcome**: Higher accuracy, precision, recall, and F1-score, illustrated in the **blue** confusion matrix.

---

## Experimental Results

### Logistic Regression (green matrix)
- Achieved ~77% overall accuracy.  
- Slightly weaker precision/recall for certain classes, especially under class imbalance.

### DistilBERT (blue matrix)
- Demonstrates improved metrics (~78–79% overall).  
- Better contextual understanding, resulting in higher precision/recall across categories.

---

**Enjoy exploring the code and results!** If you have any questions or suggestions, please open an issue or submit a pull request.
