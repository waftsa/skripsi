import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from bs4 import BeautifulSoup
from pathlib import Path

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Setup tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Title & description
st.title("🧹 Text Preprocessing")
st.markdown("Halaman ini menampilkan proses pembersihan teks secara bertahap dari data mentah tweet.")

# Load dataset
try:
    BASE_DIR = Path(__file__).resolve().parent.parent  # naik 1 folder dari /pages ke root
    data_path = BASE_DIR / "data" / "data_hasil_scraping.csv"

    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("❌ Dataset tidak ditemukan. Harap pastikan file `dataset_twitter.csv` ada di direktori.")
    st.stop()

# Tampilkan data mentah
st.subheader("📄 Data Mentah (Kolom `full_text`)")
st.dataframe(df[['full_text']].head(10))

# Simpan hanya kolom teks
texts = df['full_text'].astype(str)

# Tahap 1: Case Folding
def case_folding(text):
    return text.lower()

case_folded = texts.apply(case_folding)

with st.expander("1️⃣ Case Folding (Huruf Kecil Semua)"):
    st.dataframe(pd.DataFrame({'case_folded': case_folded.head(10)}))

# Tahap 2: Menghapus HTML tags
def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

html_cleaned = case_folded.apply(remove_html)

with st.expander("2️⃣ Menghapus Tag HTML"):
    st.dataframe(pd.DataFrame({'html_cleaned': html_cleaned.head(10)}))

# Tahap 3: Menghapus simbol, tanda baca, angka, dan mention
def remove_symbols_numbers_mentions(text):
    text = re.sub(r'@\w+', '', text)              # Hapus mention (@username)
    text = re.sub(r'[^\w\s]', '', text)           # Hapus simbol dan tanda baca
    text = re.sub(r'\d+', '', text)               # Hapus angka
    return text

symbols_cleaned = html_cleaned.apply(remove_symbols_numbers_mentions)

with st.expander("3️⃣ Menghapus Simbol, Tanda Baca, Angka, dan Mention"):
    st.dataframe(pd.DataFrame({'symbols_cleaned': symbols_cleaned.head(10)}))

# Tahap 4: Menghapus stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

stopwords_removed = symbols_cleaned.apply(remove_stopwords)

with st.expander("4️⃣ Menghapus Stopwords (English)"):
    st.dataframe(pd.DataFrame({'no_stopwords': stopwords_removed.head(10)}))

# Tahap 5: Lemmatization
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

lemmatized = stopwords_removed.apply(lemmatize_text)

with st.expander("5️⃣ Lemmatization"):
    st.dataframe(pd.DataFrame({'lemmatized': lemmatized.head(10)}))

# # Tahap 6: Stemming
# def stem_text(text):
#     return ' '.join([stemmer.stem(word) for word in text.split()])

# final_clean = lemmatized.apply(stem_text)

# with st.expander("6️⃣ Stemming"):
#     st.dataframe(pd.DataFrame({'stemmed_final': final_clean.head(10)}))

# 🔍 Tes Preprocessing dari Input User
st.subheader("🧪 Uji Coba Text Preprocessing")
input_text = st.text_area("Masukkan kalimat (berbahasa Inggris):", height=100)

if st.button("🔄 Preprocess Teks"):
    if not input_text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        # Jalankan semua tahap preprocessing
        text = case_folding(input_text)
        text = remove_html(text)
        text = remove_symbols_numbers_mentions(text)
        text = remove_stopwords(text)
        text = lemmatize_text(text)
        # text = stem_text(text)

        st.success("Hasil Preprocessing:")
        st.code(text)
