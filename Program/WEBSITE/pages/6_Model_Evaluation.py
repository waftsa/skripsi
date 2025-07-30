import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

st.title("📊 Model Comparison")
st.write("Halaman ini menampilkan perbandingan performa berbagai model yang digunakan untuk training analisis sentimen dataset.")

# Path data
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "model_score.csv"

if os.path.exists(file_path):
    df_scores = pd.read_csv(file_path)

    # Dropdown untuk memilih Split Type (unit)
    split_options = ["Semua"] + df_scores["Split Type"].unique().tolist()
    selected_split = st.selectbox("Pilih jenis Split Type:", split_options)

    # Filter berdasarkan pilihan Split Type
    if selected_split != "Semua":
        df_scores_filtered = df_scores[df_scores["Split Type"] == selected_split]
    else:
        df_scores_filtered = df_scores.copy()

    # Multiselect untuk memilih model
    available_models = df_scores_filtered["Model"].unique().tolist()
    selected_models = st.multiselect(
        "Pilih model yang ingin dibandingkan:",
        available_models,
        default=available_models
    )

    # Filter model
    final_df = df_scores_filtered[df_scores_filtered["Model"].isin(selected_models)]

    # tabel
    st.subheader("📋 Tabel Perbandingan")
    st.dataframe(final_df.set_index("Model"))

    # bar chart
    st.subheader("📈 Visualisasi Metrik")
    metric_to_plot = st.selectbox("Pilih metrik:", ["Akurasi", "Precision", "Recall", "F1-score"])

    fig, ax = plt.subplots()
    ax.bar(final_df["Model"], final_df[metric_to_plot], color="steelblue")
    ax.set_ylabel(metric_to_plot)
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric_to_plot} Comparison")
    st.pyplot(fig)

    # best model
    if not final_df.empty:
        best_model = final_df.sort_values(by=metric_to_plot, ascending=False).iloc[0]["Model"]
        st.success(f"Model dengan {metric_to_plot} tertinggi adalah **{best_model}**.")
else:
    st.error(f"Gagal memuat data. Pastikan file `{file_path}` ada dan berada di folder yang tepat.")
