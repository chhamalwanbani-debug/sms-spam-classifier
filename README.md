# SMS Spam Classifier

A machine learning web application that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP), TF-IDF feature extraction, and a Linear Support Vector Classifier (LinearSVC).

The application provides a Streamlit interface where users can enter a message and receive a spam or ham prediction.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Natural Language Processing:** NLTK
- **Feature Extraction:** TF-IDF Vectorizer
- **Classification Algorithm:** LinearSVC
- **Web Framework:** Streamlit

## 🧹 Data Preprocessing

The SMS messages undergo the following preprocessing steps:

1. **Lowercasing:** Converts text into lowercase.
2. **Tokenization:** Splits messages into individual words.
3. **Filtering:** Retains alphanumeric tokens and removes English stopwords.
4. **Stemming:** Uses Porter Stemmer to reduce words to their root forms.

## 📊 Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts preprocessed text into numerical feature vectors.

The final vectorizer uses a maximum of **5,000 features**, with the LinearSVC model configured using `C=1.0`.

Precision, recall, and F1-score are reported for the Spam class, treating Spam as the positive class.

## 🤖 Model Evaluation

The final model was selected using validation performance and evaluated using a separate held-out test set.

### 5-Fold Cross-Validation

| Metric | Mean Score |
|---|---:|
| Accuracy | 98.24% |
| Precision | 97.31% |
| Recall | 88.56% |
| F1-Score | 92.71% |

### Held-Out Test Set

| Metric | Score |
|---|---:|
| Accuracy | 98.84% |
| Precision | 97.60% |
| Recall | 93.13% |
| F1-Score | 95.31% |

*Note: Cross-validation and held-out test results are reported separately because they come from different evaluation procedures.*

## 🚀 Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/chhamalwanbani-debug/sms-spam-classifier.git
   ```

2. Navigate to the project directory:

   ```bash
   cd sms-spam-classifier
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```text
sms-spam-classifier/
│
├── app.py                   # Streamlit web application
├── evaluate.py              # Model evaluation and training
├── compare_cv.py             # Cross-validation comparison
├── sms-spam-detection.ipynb  # Exploratory data analysis
├── spam.csv                  # SMS dataset
├── model.pkl                 # Trained LinearSVC model
├── vectorizer.pkl            # TF-IDF vectorizer
├── requirements.txt          # Project dependencies
└── Dockerfile                # Docker configuration
```

## ⚠️ Limitations

- Model performance depends on the patterns represented in the training dataset.
- New, multilingual, or previously unseen spam patterns may be misclassified.
- Predictions do not guarantee that a message is safe.

## 🔮 Future Improvements

- Expand the training dataset with newer spam examples.
- Improve detection of previously unseen spam patterns.
- Explore additional classification algorithms and preprocessing techniques.
- Improve deployment and application monitoring.
