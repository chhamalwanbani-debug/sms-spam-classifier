# Real-Time SMS Spam Classifier 🧠🤖

A productionized machine learning web application engineered to perform binary (2-class) spam classification on message text in real time. This project implements a custom data engineering text preprocessing pipeline and hosts the predictive engine on a publicly accessible cloud interface.

## 🛠️ Data Stack & Engineering Pipeline

- **Core Language:** Python
- **Machine Learning Architecture:** Scikit-learn (Binary Classification Engine)
- **Natural Language Processing:** NLTK (Natural Language Toolkit)
- **Deployment Platform:** Streamlit Web Framework & Streamlit Cloud

## 🚀 NLP & Data Preprocessing Core

To transform unstructured message strings into highly optimized numeric inputs for model evaluation, raw input data passes through a structured 3-stage NLP preprocessing pipeline:
1. **Tokenization:** Breaking down messages into individual textual units (words).
2. **Standardization:** Stripping out operational noise, including system punctuation marks and standard English stopwords.
3. **Morphological Stemming:** Executing Porter Stemming via `NLTK` to reduce words to their base or root forms, drastically improving model generalization.

## 📊 Feature Extraction & Deployed Execution

- **Vectorization:** Clean text vectors are generated using a TF-IDF (Term Frequency-Inverse Document Frequency) algorithm to mathematically weigh signature keyword importance.
- **Cloud Deployment:** The finalized binary classification architecture is pushed directly onto Streamlit Cloud, delivering an instant, serverless web interface accessible with zero local framework installations required.
