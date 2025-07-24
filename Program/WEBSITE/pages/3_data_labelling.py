import streamlit as st
import pandas as pd
from pathlib import Path

st.title("📌 Data Labelled")
st.markdown("""
Halaman ini menampilkan dua jenis data berlabel:
- **Pelabelan manual** oleh peneliti.
- **Pelabelan pihak ketiga / annotator** sebagai bentuk validasi kolaboratif.
""")

# Path ke file CSV 
BASE_DIR = Path(__file__).resolve().parent.parent  # naik 1 folder dari /pages ke root
manual_path = BASE_DIR / "data" / "data_labelled_manual.csv"


annotator_path = "./dataset/label_annotator.csv"

# Tampilkan data manual
try:
    df_manual = pd.read_csv(manual_path, delimiter=";")

    with st.expander("📝 Data Label Manual"):
        st.dataframe(df_manual.head(20))

except FileNotFoundError:
    st.error(f"File data manual tidak ditemukan di path: `{manual_path}`")

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca file manual: {e}")

# Tampilkan data annotator
try:
    df_annotator = pd.read_csv(annotator_path)

    with st.expander("👥 Data Label dari Annotator"):
        st.dataframe(df_annotator.head(20))

except FileNotFoundError:
    st.error(f"File data annotator tidak ditemukan di path: `{annotator_path}`")

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca file annotator: {e}")
