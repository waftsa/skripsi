import streamlit as st
import pandas as pd
from pathlib import Path

# Judul halaman
st.title("📊 Dataset Penelitian")

# Penjelasan dataset
st.markdown("""
Dataset ini diambil dari Twitter dengan menggunakan beberapa kata kunci utama, yaitu:

- **'genocide'**
- **'gaza'**
- **'palestine'**
- **'israel'**

Namun, mengingat adanya pemblokiran atau pembatasan visibilitas terhadap beberapa kata sensitif di platform, pengambilan data juga dilakukan dengan memanfaatkan variasi kata yang disensor atau dimodifikasi oleh pengguna, antara lain:

- **pa1estine**
- **pa|estine**
- **p@lestine**
- **plstn**
- **izrael**
- **isr@el**
- **1srael**
- **g@za**
- **gz@**
- **gen0cide**

Scraping dilakukan pada rentang waktu **7 Oktober 2023 hingga 7 April 2025**, bertepatan dengan periode meningkatnya eskalasi konflik antara Israel dan Palestina. Dataset ini digunakan sebagai data latih untuk model analisis sentimen.
""")


BASE_DIR = Path(__file__).resolve().parent.parent  # naik 1 folder dari /pages ke root
data_path = BASE_DIR / "data" / "data_hasil_scraping.csv"

df = pd.read_csv(data_path)

# Tampilkan semua data
st.subheader("📌 Seluruh Data yang Terkumpul")
st.dataframe(df)

# Tampilkan data berdasarkan keyword
st.subheader("🔍 Filter Data Berdasarkan Kata Kunci")

keywords = [
    'genocide', 'gen0cide',
    'gaza', 'g@za', 'gz@',
    'palestine', 'pa1estine', 'pa|estine', 'p@lestine', 'plstn',
    'israel', 'izrael', 'isr@el', '1srael'
]

selected_keywords = st.multiselect("Pilih Kata Kunci", sorted(set(keywords)))

if selected_keywords:
    pattern = '|'.join(selected_keywords)
    filtered = df[df['full_text'].str.contains(pattern, case=False, na=False)]
    st.markdown(f"**Menampilkan data untuk keyword : `{', '.join(selected_keywords)}`**")
    st.dataframe(filtered)
else:
    st.info("Silahkan pilih satu atau lebih keyword untuk menampilkan data.")