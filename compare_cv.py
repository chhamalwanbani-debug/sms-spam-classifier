
import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].drop_duplicates()
df.columns = ['target', 'text']

df['target'] = df['target'].map({'ham': 0, 'spam': 1})

ps = PorterStemmer()
sw = set(stopwords.words('english'))

def transform_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [
        ps.stem(w) for w in words
        if w.isalnum() and w not in sw
    ]
    return ' '.join(words)

X = df['text'].apply(transform_text)
y = df['target']

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for features in [5000, 10000]:

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=features,
            ngram_range=(1, 2)
        )),
        ('model', LinearSVC(C=1))
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1']
    )

    print('\nFEATURES:', features)

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = scores['test_' + metric]
        print(
            metric.upper(),
            'Mean:', round(values.mean(), 4),
            'Std:', round(values.std(), 4)
        )

print('\nCROSS-VALIDATION COMPLETED')
