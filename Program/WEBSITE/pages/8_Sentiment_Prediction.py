import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from pathlib import Path
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# Judul halaman
st.header("🔮 Sentiment Prediction")
st.write("Masukkan teks di bawah ini untuk memprediksi sentimen menggunakan model Machine Learning yang telah dilatih.")

# Path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_LSTM = BASE_DIR / "model" / "lstm_model.keras"
MODEL_BILSTM = BASE_DIR / "model" / "bilstm_model.keras"
MODEL_NB = BASE_DIR / "model" / "naive_bayes_model.pkl"
MODEL_SVM = BASE_DIR / "model" / "svm_model.pkl"
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"

# Load model dan tokenizer
@st.cache_resource
def load_keras_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_pickle_model(path):
    with open(path, "rb") as f:
        return joblib.load(f)
    
@st.cache_resource
def load_vectorizer(path):
    with open(path, "rb") as f:
        return joblib.load(f)
    
vectorizer = load_vectorizer(VECTORIZER_PATH)


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()

#load all model
model_lstm = load_keras_model(MODEL_LSTM)
model_bilstm = load_keras_model(MODEL_BILSTM)
model_nb = load_pickle_model(MODEL_NB)
model_svm = load_pickle_model(MODEL_SVM)

# Input teks
text_input = st.text_area("Masukkan kalimat yang ingin diprediksi:", height=150)

# Fungsi preprocessing 
stop_words = set(stopwords.words('english'))
sentiment_stopwords = {
    "the", "a", "an", "and", "or", "but", "if", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "you", "i", "me", "my", "we", "us", "our", "they",
    "them", "their", "he", "him", "his", "she", "her", "it", "its", "this", "that", "these", "those"
}
custom_stopwords = stop_words - sentiment_stopwords

stemmer = PorterStemmer()
sensitive_words = {"israel", "palestine", "hamas", "genocide", "gaza", "zionist"}
lemmatizer = WordNetLemmatizer()

def custom_stem_id(word):
    word_lower = word.lower()
    if word_lower in sensitive_words:
        return word_lower
    else:
        lemma = lemmatizer.lemmatize(word_lower)
        return stemmer.stem(lemma)

def text_preprocess(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)      # hapus URL
    text = re.sub(r"@\w+", "", text)                         # hapus mention
    text = re.sub(r"[^\w\s]", "", text)                      # hapus tanda baca
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)      # hapus emoji
    text = re.sub(r"\d+", "", text)                          # hapus angka
    text = re.sub(r"\s+", " ", text).strip()                 # hapus spasi berlebih
    text = re.sub(r'[^\x00-\x7F]+', '', str(text))           # hapus simbol non-ASCII
    words = word_tokenize(text)
    words = [word for word in words if word not in custom_stopwords]
    stemmed_tokens = [custom_stem_id(w) for w in words]
    return " ".join(stemmed_tokens)

def text_tokenizing(text, tokenizer, max_len=100):
    cleaned = text_preprocess(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

#funsi prediction
def predict_lstm(text, model, tokenizer):
    padded = text_tokenizing(text, tokenizer)
    pred = model.predict(padded)[0]
    pred = np.array(pred).flatten()  

    if len(pred) == 1:
        label = "Pro-Genocide" if pred[0] > 0.5 else "Cons-Genocide"
        confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
    else:
        labels = ["Cons-Genocide", "Netral", "Pro-Genocide"]
        label_idx = np.argmax(pred)
        label = labels[label_idx]
        confidence = pred[label_idx]
    return label, confidence


def predict_sklearn(text, model, vectorizer):
    cleaned = text_preprocess(text)
    X = vectorizer.transform([cleaned])  
    try:
        probs = model.predict_proba(X)[0]
        label_idx = np.argmax(probs)
        confidence = probs[label_idx]
    except AttributeError:
        label_idx = model.predict(X)[0]
        confidence = 1.0

    labels = ["Cons-Genocide", "Netral", "Pro-Genocide"]
    label = labels[label_idx] if label_idx < len(labels) else str(label_idx)
    return label, confidence

# Tombol prediksi
if st.button("Prediksi"):
    if text_input.strip() == "":
        st.warning("Mohon masukkan kalimat terlebih dahulu.")
    else:
        results = {}

        # Model LSTM
        label_lstm, conf_lstm = predict_lstm(text_input, model_lstm, tokenizer)
        results["LSTM"] = (label_lstm, conf_lstm)

        # Model BiLSTM
        label_bilstm, conf_bilstm = predict_lstm(text_input, model_bilstm, tokenizer)
        results["BiLSTM"] = (label_bilstm, conf_bilstm)

        # Model Naive Bayes
        label_nb, conf_nb = predict_sklearn(text_input, model_nb, vectorizer)
        results["Naive Bayes"] = (label_nb, conf_nb)

        # Model SVM
        label_svm, conf_svm = predict_sklearn(text_input, model_svm, vectorizer)
        results["SVM"] = (label_svm, conf_svm)

        # Menentukan model dengan confidence tertinggi
        best_model = max(results.items(), key=lambda x: x[1][1])

        # Tampilkan hasil
        st.subheader("📊 Hasil Prediksi Tiap Model")
        for model_name, (label, conf) in results.items():
            st.write(f"**{model_name}:** **{label}** — **{conf*100:.2f}%**")

        st.success(f"**Model dengan Nilai Confidence Tertinggi :** {best_model[0]} — {best_model[1][0]} ({best_model[1][1]*100:.2f}%)")

