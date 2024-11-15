import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
def lower_case(text):
    return text.lower()

def tokenization(text):
    return nltk.word_tokenize(text)

def remove_special_characters(text):
    x = []
    for i in text:
        if i.isalnum():
            x.append(i)
    text = x[:]
    return text

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    x=[]
    for i in text:
        if i not in stop_words:
            x.append(i)
    text = x[:]
    return text

def punctuators(text):
    x=[]
    for i in text:
        if i not in string.punctuation:
            x.append(i)
    text = x[:]
    return text

def stemming(text):
    ps = PorterStemmer()
    x = []
    for i in text:
        x.append(ps.stem(i))
    text = x[:]
    return " ".join(text)
def transform_text(text):
    text = lower_case(text)
    text = tokenization(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    text = punctuators(text)
    text = stemming(text)
    return text
with open('vectorizer.pkl','rb') as file:
    tfidf = pickle.load(file)
with open('model.pkl','rb') as file1:
    model = pickle.load(file1)
st.title("Email Spam Classifier📧")
input_text = st.text_area("Enter the Email text")
if st.button("Predict"):
    transformed_text = transform_text(input_text)
    vectorized_text = tfidf.transform([transformed_text])
    result = model.predict(vectorized_text)[0]
    if result==1:
        st.header("Spam 🚨")
    else:
        st.header("Not Spam 😊")
