
import pandas as pd
import numpy as np
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Clean dataset
df = df[['v1', 'v2']].copy()
df.columns = ['target', 'text']
df = df.drop_duplicates().reset_index(drop=True)

# Encode labels
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['target'].value_counts())

# Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    tokens = nltk.word_tokenize(str(text).lower())

    tokens = [
        word for word in tokens
        if word.isalnum()
        and word not in stop_words
        and word not in string.punctuation
    ]

    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

print("\nPreprocessing messages...")
df['transformed_text'] = df['text'].apply(transform_text)

# Split: 60% train, 20% validation, 20% test
X = df['transformed_text']
y = df['target']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=2,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=2,
    stratify=y_train_val
)

print("\nData split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# Fit vectorizer ONLY on training data
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)

# Compare models using validation data
models = {
    'MultinomialNB': MultinomialNB(alpha=0.1),
    'LinearSVC': LinearSVC(C=1.0, max_iter=5000),
    'LogisticRegression': LogisticRegression(
        C=5.0,
        max_iter=5000
    )
}

best_model_name = None
best_f1 = -1

print("\nVALIDATION RESULTS")

for name, model in models.items():

    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_val_vec)

    acc = accuracy_score(y_val, predictions)
    prec = precision_score(y_val, predictions, zero_division=0)
    rec = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(f"\n{name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

# Refit vectorizer on training + validation data
final_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_final = final_vectorizer.fit_transform(X_train_val)
X_test_final = final_vectorizer.transform(X_test)

# Train selected model on training + validation data
final_models = {
    'MultinomialNB': MultinomialNB(alpha=0.1),
    'LinearSVC': LinearSVC(C=1.0, max_iter=5000),
    'LogisticRegression': LogisticRegression(
        C=5.0,
        max_iter=5000
    )
}

final_model = final_models[best_model_name]
final_model.fit(X_train_final, y_train_val)

# Evaluate once on untouched test data
y_pred = final_model.predict(X_test_final)

print("\n" + "=" * 45)
print("FINAL TEST RESULTS")
print("=" * 45)

print("Selected model:", best_model_name)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Ham', 'Spam'],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Save final model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(final_vectorizer, f)

print("\nFinal model and vectorizer saved successfully!")
