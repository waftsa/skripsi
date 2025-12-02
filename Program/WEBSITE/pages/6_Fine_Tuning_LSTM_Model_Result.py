import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Judul Halaman ---
st.header("🔧 LSTM Hyperparameter Fine-Tuning Dashboard")
st.write("Gunakan filter di bawah untuk melihat hasil eksperimen berdasarkan kombinasi hyperparameter.")

# --- Load Data ---
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "data" / "lstm_tuning_results.csv"
df = pd.read_csv(csv_path)


# =========================
# 🎛️ FILTER SECTION
# =========================

st.subheader("🎚️ Filter Hyperparameter")

col1, col2, col3 = st.columns(3)

with col1:
    selected_emb = st.multiselect(
        "Embedding Dim",
        options=sorted(df["Embedding Dim"].unique()),
        default=sorted(df["Embedding Dim"].unique())   # default: select all
    )

with col2:
    selected_units = st.multiselect(
        "Units",
        options=sorted(df["Units"].unique()),
        default=sorted(df["Units"].unique())
    )

with col3:
    selected_dropout = st.multiselect(
        "Dropout",
        options=sorted(df["Dropout"].unique()),
        default=sorted(df["Dropout"].unique())
    )

col4, col5, col6 = st.columns(3)

with col4:
    selected_batch = st.multiselect(
        "Batch Size",
        options=sorted(df["Batch Size"].unique()),
        default=sorted(df["Batch Size"].unique())
    )

with col5:
    selected_lr = st.multiselect(
        "Learning Rate",
        options=sorted(df["Learning Rate"].unique()),
        default=sorted(df["Learning Rate"].unique())
    )

with col6:
    selected_epoch = st.multiselect(
        "Epoch",
        options=sorted(df["Epoch"].unique()),
        default=sorted(df["Epoch"].unique())
    )

# =========================
# 🧮 FILTER DATAFRAME
# =========================

df_filtered = df[
    (df["Embedding Dim"].isin(selected_emb)) &
    (df["Units"].isin(selected_units)) &
    (df["Dropout"].isin(selected_dropout)) &
    (df["Batch Size"].isin(selected_batch)) &
    (df["Learning Rate"].isin(selected_lr)) &
    (df["Epoch"].isin(selected_epoch))
]


st.subheader("📄 Hasil Fine-Tuning berdasarkan Filter")
st.dataframe(df_filtered)


if not df_filtered.empty:
    best_row = df_filtered.sort_values(by="Accuracy", ascending=False).iloc[0]
    st.success(f"🏆 Model terbaik setelah filter: **Accuracy {best_row['Accuracy']:.4f}**")
    st.json(best_row.to_dict())
else:
    st.warning("Tidak ada data yang cocok dengan filter.")

st.subheader("📈 Grafik Akurasi (Filtered)")
if not df_filtered.empty:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_filtered["Accuracy"].values, marker="o")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy for Selected Hyperparameters")
    st.pyplot(fig)

st.subheader("🔍 Korelasi Hyperparameter (Filtered)")
if df_filtered.shape[0] > 1:
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(df_filtered.corr(numeric_only=True), annot=True, cmap="Blues")
    st.pyplot(fig2)
