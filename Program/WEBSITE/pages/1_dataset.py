import streamlit as st
import pandas as pd
from pathlib import Path

# Judul halaman
st.title("📊 Dataset Penelitian")

# Penjelasan dataset
st.markdown("""
Dataset ini diambil dari Twitter dengan menggunakan beberapa kata kunci, yaitu:

- **'genocide'**
- **'gaza'**
- **'palestine'**
- **'israel'**

Pengambilan data ini dimulai dari tanggal 7 oktober 2023 hingga 7 april 2025. Dataset tersebut akan digunakan sebagai data untuk training model.
""")
BASE_DIR = Path(__file__).resolve().parent.parent  # naik 1 folder dari /pages ke root
data_path = BASE_DIR / "data" / "data_hasil_scraping.csv"

df = pd.read_csv(data_path)

# Tampilkan semua data
st.subheader("📌 Seluruh Data yang Terkumpul")
st.dataframe(df)

# Tampilkan data berdasarkan keyword
st.subheader("🔍 Filter Data Berdasarkan Kata Kunci")

keywords = ['genocide', 'gaza', 'palestine', 'israel']

for keyword in keywords:
    st.markdown(f"**Keyword: {keyword}**")
    filtered = df[df['full_text'].str.contains(keyword, case=False, na=False)]
    st.dataframe(filtered)
