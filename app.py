import streamlit as st
import pickle
import string
import nltk
import warnings
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

warnings.filterwarnings('ignore')

# Download NLTK data to /tmp (guaranteed writable on Streamlit Cloud)
import os
NLTK_DATA_DIR = '/tmp/nltk_data'
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
os.environ['NLTK_DATA'] = NLTK_DATA_DIR
nltk.data.path.insert(0, NLTK_DATA_DIR)

nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('punkt_tab', download_dir=NLTK_DATA_DIR)
nltk.download('stopwords', download_dir=NLTK_DATA_DIR)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# Load the models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms]).toarray()
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. display
        if result == 1:
            st.error("Spam Detected")
        else:
            st.success("Not Spam")
