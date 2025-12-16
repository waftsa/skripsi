import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.header("📊 Evaluasi Model Klasifikasi Sentimen")
st.markdown("Visualisasi hasil tuning dan evaluasi model machine learning dan deep learning")

BASE_DIR = Path(__file__).resolve().parent.parent

EVAL_FILES = {
    "Naive Bayes": BASE_DIR / "data/evaluasi_nb.csv",
    "SVM": BASE_DIR / "data/evaluasi_svm.csv",
    "LSTM": BASE_DIR / "data/evaluasi_lstm.csv",
    "BiLSTM": BASE_DIR / "data/evaluasi_bilstm.csv",
}

IMAGE_FILES = {
    "Naive Bayes": {
        "cm": BASE_DIR / "image/nb_confusion.png",
        "roc": BASE_DIR / "image/nb_roc.png",
    },
    "SVM": {
        "cm": BASE_DIR / "image/svm_confusion.png",
        "roc": BASE_DIR / "image/svm_roc.png",
    },
    "LSTM": {
        "cm": BASE_DIR / "image/lstm_confusion.png",
        "roc": BASE_DIR / "image/lstm_roc.png",
    },
    "BiLSTM": {
        "cm": BASE_DIR / "image/bilstm_confusion.png",
        "roc": BASE_DIR / "image/bilstm_roc.png",
    },
}

PARAMETERS = {
    "Naive Bayes": {
        "alpha": "0.1",
        "fit_prior": "False",
        "TF-IDF Norm": "l1"
    },
    "SVM": {
        "Kernel": "RBF",
        "C": "1",
        "Gamma": "Scale",
        "TF-IDF Norm": "l2"
    },
    "LSTM": {
        "Embedding Dim": "100",
        "Units": "32",
        "Dropout": "0.2",
        "Batch Size": "16",
        "Learning Rate": "0.0001",
        "Epoch": "25"
    },
    "BiLSTM": {
        "Merge Mode": "Concat",
        "Embedding Dim": "100",
        "Units": "32",
        "Dropout": "0.2",
        "Batch Size": "16",
        "Learning Rate": "0.0001",
        "Epoch": "25"
    }
}

model_choice = st.selectbox(
    "Pilih Model:",
    list(EVAL_FILES.keys())
)

# PARAMETER

st.subheader("⚙️ Parameter Optimal")

param_df = pd.DataFrame(
    PARAMETERS[model_choice].items(),
    columns=["Parameter", "Nilai Optimal"]
)
st.table(param_df)

# TABEL EVALUASI

st.subheader("📌 Hasil Evaluasi Model")

df_eval = pd.read_csv(EVAL_FILES[model_choice])
st.dataframe(df_eval)

st.markdown("**Rata-rata Evaluasi:**")
st.write(df_eval.mean(numeric_only=True))

st.subheader("📈 Visualisasi Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Confusion Matrix**")
    st.image(Image.open(IMAGE_FILES[model_choice]["cm"]))

with col2:
    st.markdown("**ROC Curve**")
    st.image(Image.open(IMAGE_FILES[model_choice]["roc"]))
