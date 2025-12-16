import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent

# load Modul
@st.cache_resource
def load_word2vec():
    return Word2Vec.load(str(BASE_DIR / "model" / "word2vec_full.model"))

@st.cache_resource
def load_tfidf():
    vectorizer = joblib.load(BASE_DIR / "model" / "tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load(BASE_DIR / "model" / "tfidf_matrix.pkl")
    return vectorizer, tfidf_matrix

@st.cache_data
def load_corpus():
    corpus = joblib.load(BASE_DIR / "model" / "corpus.pkl")
    return corpus

w2v_model = load_word2vec()
tfidf_vectorizer, tfidf_matrix = load_tfidf()
corpus = load_corpus()

st.header("🧠 Ekstraksi Fitur Teks")
st.markdown("Hasil ekstraksi fitur menggunkan TF-IDF dan Word2Vec")
# select method
method = st.selectbox(
    "🔧 Pilih Metode Ekstraksi Fitur",
    ["Word2Vec", "TF-IDF"]
)

# word2vec
if method == "Word2Vec":
    st.subheader("🔍 Word2Vec")

    mode = st.radio(
        "Pilih mode:",
        ["Cari Similarity", "Lihat Vektor Kata"]
    )

    input_word = st.text_input("Masukkan kata:")

    if input_word:
        if input_word not in w2v_model.wv:
            st.warning(f"Kata '{input_word}' tidak ditemukan di vocabulary.")
        else:
            if mode == "Cari Similarity":
                similar_words = w2v_model.wv.most_similar(input_word, topn=10)
                df_sim = pd.DataFrame(similar_words, columns=["Kata", "Similarity"])
                st.table(df_sim)
            if mode == "Lihat Vektor Kata":
                vector = w2v_model.wv[input_word]
                df_vector = pd.DataFrame(
                    vector.reshape(1, -1),
                    columns=[f"dim_{i+1}" for i in range(len(vector))]
                )
                st.dataframe(df_vector)

    st.subheader("📌 Contoh Vektor Kata")
    words = w2v_model.wv.index_to_key[:10]
    vectors = [w2v_model.wv[word] for word in words]
    df_vectors = pd.DataFrame(vectors, index=words)
    st.dataframe(df_vectors)

# TF-IDF
if method == "TF-IDF":
    st.subheader("📄 TF-IDF – Kemiripan & Bobot Kata")

    input_text = st.text_area("Masukkan kalimat:")

    if input_text:
        input_vec = tfidf_vectorizer.transform([input_text])
        similarity = cosine_similarity(input_vec, tfidf_matrix)[0]

        st.markdown("### 🔁 Kemiripan dengan Dokumen")
        df_sim = pd.DataFrame({
            "Dokumen": corpus,
            "Similarity": similarity
        }).sort_values(by="Similarity", ascending=False)

        st.table(df_sim.head(5))

        st.markdown("### 📌 Bobot TF-IDF Kata")

        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = input_vec.toarray()[0]

        df_weights = pd.DataFrame({
            "Kata": feature_names,
            "Bobot TF-IDF": tfidf_scores
        })

        df_weights = df_weights[df_weights["Bobot TF-IDF"] > 0]
        df_weights = df_weights.sort_values(
            by="Bobot TF-IDF", ascending=False
        )

        st.table(df_weights.head(10))
