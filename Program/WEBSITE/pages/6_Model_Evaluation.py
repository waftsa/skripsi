import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# --- Judul Halaman ---
st.header("📊 Model Evaluation & Comparison")
st.write("""
Halaman ini menampilkan evaluasi performa tiap model **(Naive Bayes, SVM, LSTM, BiLSTM)** 
serta perbandingan antar model berdasarkan metrik utama.
""")

# --- Path file CSV ---
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "model_score.csv"

# --- Cek ketersediaan file ---
if os.path.exists(file_path):
    df_scores = pd.read_csv(file_path)

    # --- Pilih Model ---
    model_list = df_scores["Model"].unique().tolist()
    selected_model = st.selectbox("Pilih model untuk evaluasi detail:", model_list)

    # Filter data untuk model terpilih
    model_data = df_scores[df_scores["Model"] == selected_model]

    st.subheader(f"📈 Evaluasi Detail: {selected_model}")
    st.write("Berikut metrik performa untuk model yang dipilih:")

    # Tabel metrik (bisa jadi 1 atau beberapa split type)
    st.dataframe(model_data.set_index("Split Type"))

    # --- Confusion Matrix ---
    st.subheader("🧩 Confusion Matrix")

    # Path confusion matrix
    cm_path = BASE_DIR / "data" / f"confusion_{selected_model.lower().replace('-', '_')}.csv"
    if os.path.exists(cm_path):
        cm = pd.read_csv(cm_path, index_col=0)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {selected_model}")
        st.pyplot(fig)
    else:
        st.warning(f"Confusion matrix untuk **{selected_model}** belum tersedia di `{cm_path.name}`.")

    st.markdown("---")
    st.subheader("⚖️ Model Comparison")
    st.write("Bagian ini membandingkan semua model berdasarkan metrik yang dipilih.")

    # Pilih split type
    split_options = ["Semua"] + df_scores["Split Type"].unique().tolist()
    selected_split = st.selectbox("Pilih Split Type untuk perbandingan:", split_options)

    if selected_split != "Semua":
        df_filtered = df_scores[df_scores["Split Type"] == selected_split]
    else:
        df_filtered = df_scores.copy()

    # Pilih metrik untuk chart
    metric = st.selectbox("Pilih metrik untuk perbandingan:", ["Akurasi", "Precision", "Recall", "F1-score"])

    fig, ax = plt.subplots()
    ax.bar(df_filtered["Model"], df_filtered[metric], color="teal")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(f"Perbandingan {metric} antar model")
    st.pyplot(fig)

    # Tampilkan model terbaik
    if not df_filtered.empty:
        best_model = df_filtered.sort_values(by=metric, ascending=False).iloc[0]
        st.success(f"🏆 Model dengan {metric} tertinggi adalah **{best_model['Model']} ({best_model[metric]:.2f})**")

else:
    st.error(f"❌ Gagal memuat data. Pastikan file `{file_path}` tersedia.")

