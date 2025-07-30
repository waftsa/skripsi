import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


st.title("⚖️ Data Balancing")
st.markdown("""
Halaman ini menampilkan kondisi distribusi data awal serta hasil proses balancing menggunakan:
- **Oversampling**: Menambah data pada kelas minoritas.
- **Undersampling**: Mengurangi data pada kelas mayoritas.
""")

# Path ke file
BASE_DIR = Path(__file__).resolve().parent.parent 
path_imbalanced = BASE_DIR / "data" / "dataset_imbalanced.csv"

BASE_DIR = Path(__file__).resolve().parent.parent 
path_oversampled = BASE_DIR / "data" / "dataset_oversampling.csv"

BASE_DIR = Path(__file__).resolve().parent.parent 
path_undersampled = BASE_DIR / "data" / "dataset_undersampling.csv"


# Fungsi untuk plot pie chart
def show_pie_chart(df, label, title):
    label_counts = df[label].value_counts()
    
    # Urutan label yang diinginkan untuk warna tetap konsisten
    ordered_labels = ['Contra-genocide', 'Netral', 'Pro-genocide']
    colors = {
        'Contra-genocide': 'red',
        'Netral': 'yellow',
        'Pro-genocide': 'green'
    }
    
    label_counts = label_counts.reindex(ordered_labels).dropna()
    pie_colors = [colors[lbl] for lbl in label_counts.index]

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%',
           startangle=90, colors=pie_colors)
    ax.axis('equal')
    st.pyplot(fig)
    st.caption(title)
    
    # Tampilkan tabel jumlah
    st.markdown("#### 📋 Jumlah Tiap Label")
    st.dataframe(label_counts.reset_index().rename(columns={'index': 'Label', label: 'Jumlah'}))

# Pie chart awal (imbalanced)
try:    
    df_imbalanced = pd.read_csv(path_imbalanced, delimiter=';')
    label_map = {
        'c': 'Contra-genocide', 0: 'Contra-genocide',
        'n': 'Netral',          1: 'Netral',
        'p': 'Pro-genocide',    2: 'Pro-genocide'
    }
    
    df_imbalanced['label'] = df_imbalanced['label'].map(label_map)
    
    st.subheader("📊 Distribusi Data Awal")
    show_pie_chart(df_imbalanced, label='label', title="Distribusi Label Sebelum Balancing")
    
except Exception as e:
    st.error(f"Gagal memuat data: {e}")

# Oversampling
try:
    df_over = pd.read_csv(path_oversampled, delimiter=';')

    with st.expander("📈 Oversampling"):
        label_map = {
        'c': 'Contra-genocide', 0: 'Contra-genocide',
        'n': 'Netral',          1: 'Netral',
        'p': 'Pro-genocide',    2: 'Pro-genocide'
        }
        
        df_over['label'] = df_over['label'].map(label_map)
        
        show_pie_chart(df_over, label='label', title="Distribusi Setelah Oversampling")

except Exception as e:
    st.error(f"Gagal membaca file oversampled: {e}")

# Undersampling
try:
    df_under = pd.read_csv(path_undersampled)

    with st.expander("📉 Undersampling"):
        label_map = {
        'c': 'Contra-genocide', 0: 'Contra-genocide',
        'n': 'Netral',          1: 'Netral',
        'p': 'Pro-genocide',    2: 'Pro-genocide'
        }
        
        df_under['label'] = df_under['label'].map(label_map)
        
        show_pie_chart(df_under, label='label', title="Distribusi Setelah Undersamplingsampling")

except Exception as e:
    st.error(f"Gagal membaca file undersampled: {e}")

# Pie data akhir
try:    
    df_final = pd.read_csv(path_undersampled)
    label_map = {
        'c': 'Contra-genocide', 0: 'Contra-genocide',
        'n': 'Netral',          1: 'Netral',
        'p': 'Pro-genocide',    2: 'Pro-genocide'
    }
    
    df_final['label'] = df_final['label'].map(label_map)
    
    st.subheader("📊 Distribusi Data Akhir")
    show_pie_chart(df_final, label='label', title="Distribusi Label Setelah Data Balancing")
    
except Exception as e:
    st.error(f"Gagal memuat data: {e}")