import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

##load word index and reversed word index
word_index = imdb.get_word_index()
reverse_word_index = {value : key for key,value in word_index.items()}

#load the model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "simple_rnn_imdb.keras")

model = load_model(MODEL_PATH)

##utility functions
##1.decode_review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])

##2.preprocess_text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

##prediction fn.
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment,prediction[0][0]

##streamlit setup
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie Review")
if st.button('classify'):
    sent,pred = predict_sentiment(user_input)
    st.write(f"Sentiment: {sent}")
    st.write(f"Prediction score: {pred}")
else:
    st.write("Please enter a movie review")