import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from pathlib import Path

# Judul halaman
st.title("🔮 Sentiment Prediction")
st.write("Masukkan teks di bawah ini untuk memprediksi sentimen menggunakan model LSTM yang telah dilatih.")

# Path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_LSTM = BASE_DIR / "model" / "lstm_model.keras"
TOKENIZER_PATH = BASE_DIR / "data" / "tokenizer.pkl"

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_LSTM)

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

# Input teks
text_input = st.text_area("Masukkan kalimat yang ingin diprediksi:", height=150)

# Fungsi preprocessing & prediksi
def predict_sentiment(text, tokenizer, model, max_len=100):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded)[0]

    if len(prediction) == 1:  # binary
        label = "Pro-Genocide" if prediction[0] > 0.5 else "Cons-Genocide"
        confidence = prediction[0] if prediction[0] > 0.5 else 1 - prediction[0]
    else:  # multiclass
        label_idx = np.argmax(prediction)
        labels = ["Cons-Genocide", "Netral", "Pro-Genocide"] 
        label = labels[label_idx]
        confidence = prediction[label_idx]

    return label, confidence

# Tombol prediksi
if st.button("Prediksi"):
    if text_input.strip() == "":
        st.warning("Mohon masukkan kalimat terlebih dahulu.")
    else:
        label, confidence = predict_sentiment(text_input, tokenizer, model)
        st.success(f"**Sentimen:** {label} ({confidence:.2f})")
