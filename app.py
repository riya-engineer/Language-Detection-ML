import streamlit as st
import pickle

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