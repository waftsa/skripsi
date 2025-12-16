import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

st.header("📊 Model Comparison – Sentiment Classification")
st.caption("Perbandingan akurasi, precision, recall, dan F1-score rata-rata tiap model")

BASE_DIR = Path(__file__).resolve().parent.parent

# CSV evaluasi
EVAL_FILES = {
    "Naive Bayes": BASE_DIR / "data/evaluasi_nb.csv",
    "SVM": BASE_DIR / "data/evaluasi_svm.csv",
    "LSTM": BASE_DIR / "data/evaluasi_lstm.csv",
    "BiLSTM": BASE_DIR / "data/evaluasi_bilstm.csv",
}

# Load semua rata-rata metric
comparison_data = []
for model_name, file in EVAL_FILES.items():
    df = pd.read_csv(file)
    mean_metrics = df.mean(numeric_only=True)
    comparison_data.append({
        "Model": model_name,
        "Accuracy": mean_metrics["Accuracy"],
        "F1-score": mean_metrics["F1-score"],
        "Precision": mean_metrics["Precision"],
        "Recall": mean_metrics["Recall"]
    })

df_comparison = pd.DataFrame(comparison_data)
st.subheader("📌 Tabel Perbandingan Model (Rata-rata)")
st.dataframe(df_comparison)

st.subheader("📈 Perbandingan Metric – Bar Chart")
metrics = ["Accuracy", "F1-score", "Precision", "Recall"]

fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(len(df_comparison))
width = 0.2

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, df_comparison[metric], width, label=metric)

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(df_comparison["Model"])
ax.set_ylim(0, 100)
ax.set_ylabel("Persentase (%)")
ax.set_title("Perbandingan Metric Model")
ax.legend()
st.pyplot(fig)

st.subheader("🖼️ Confusion Matrix (contoh per model)")

col1, col2, col3, col4 = st.columns(4)
IMAGE_FILES = {
    "Naive Bayes": BASE_DIR / "image/nb_confusion.png",
    "SVM": BASE_DIR / "image/svm_confusion.png",
    "LSTM": BASE_DIR / "image/lstm_confusion.png",
    "BiLSTM": BASE_DIR / "image/bilstm_confusion.png",
}

with col1:
    st.image(Image.open(IMAGE_FILES["Naive Bayes"]), caption="Naive Bayes")
with col2:
    st.image(Image.open(IMAGE_FILES["SVM"]), caption="SVM")
with col3:
    st.image(Image.open(IMAGE_FILES["LSTM"]), caption="LSTM")
with col4:
    st.image(Image.open(IMAGE_FILES["BiLSTM"]), caption="BiLSTM")
