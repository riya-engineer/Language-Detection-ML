import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Check if model files exist, if not train them
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    st.info("Training model... (first run may take a moment)")
    
    # Load dataset
    df = pd.read_csv("language.csv")
    
    # Columns: Text, Language
    X = df["Text"]
    y = df["language"]
    
    # Vectorization
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
    
    # Model
    model = MultinomialNB()
    model.fit(X_vec, y)
    
    # Save files
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("🌍 Language Detection App")

text = st.text_area("Enter text:")

if st.button("Detect Language"):
    if text.strip() != "":
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        st.success(f"Detected Language: {prediction}")
    else:
        st.warning("Please enter some text")
