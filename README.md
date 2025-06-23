# 📰 Fake News Detection using NLP and Machine Learning

This project detects fake news articles using Natural Language Processing (NLP) techniques and a Machine Learning classification model. The goal is to accurately classify whether a given news headline or article is **real** or **fake**.

## 🚀 Overview

- Preprocessed a labeled dataset of real and fake news
- Applied text cleaning, tokenization, and TF-IDF vectorization
- Trained and evaluated multiple classification models (Logistic Regression, etc.)
- Achieved high accuracy and performance metrics
- Deployed as a simple web app using **Streamlit**

---

## 📁 Dataset

The dataset contains labeled news articles from sources such as Kaggle. It includes two classes:

- **Real News**
- **Fake News**

Each sample includes:

- `title`
- `text`
- `label` (0 = Real, 1 = Fake)

Example dataset: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## 🧠 Model Pipeline

1. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation and stopwords
   - Tokenization
2. **Feature Engineering**
   - TF-IDF vectorization
3. **Model**
   - Logistic Regression (also tested Random Forest and Naive Bayes)
4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix

---

## ✅ Results

- **Accuracy**: 98.6%  
- **Precision/Recall/F1-score**: ~0.99 for both classes  
- **Confusion Matrix**: Very low false positives and negatives

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Streamlit

---

## 💻 How to Run Locally

1. **Clone this repository:**

git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector

2. **Create virtual environment (optional but recommended):**

python -m venv venv
venv\Scripts\activate    # Windows

3. **Install dependencies:**

pip install -r requirements.txt

4. **Run the app:**
 
streamlit run app.py

## 📂 Project Structure

fake-news-detector/
├── app.py                  # Streamlit application
├── fake_news_model.pkl     # Trained ML model
├── vectorizer.pkl          # TF-IDF vectorizer
├── dataset.csv             # Labeled dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project overview

## 🧪 Sample Predictions
You can enter any custom news text into the app and it will classify it as:

✅ Real

❌ Fake

## 🔮 Future Work
Add BERT-based deep learning model

Collect live data using news APIs

Interpret results with LIME/SHAP

Deploy on Hugging Face Spaces or Heroku

---

Let me know if you'd like a version for **Colab**, or want to include screenshots or demo video links. I can also help write a good LinkedIn post to go with this!

## 🙋‍♂️ Author
LinkedIn: https://www.linkedin.com/in/akash-rajput-1a25b52a0/
For academic and personal learning purposes 🎓

⭐ Show Your Support
If you find this project helpful, consider giving it a ⭐ on GitHub!
