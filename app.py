import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Paste a news article or headline below to check if it's Real or Fake.")

# Input text box
user_input = st.text_area("Enter News Content", height=200)

# Predict button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.subheader("Prediction:")
        st.success(label)
