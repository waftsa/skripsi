import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from pathlib import Path

# model path 
BASE_DIR = Path(__file__).resolve().parent.parent 
path_model = BASE_DIR / "model" / "word2vec_full.model"

# load model
@st.cache_resource
def load_model():
    model = Word2Vec.load(str(path_model))
    return model

model = load_model()

# judul
st.title("🧠 Word Embedding - Word2Vec")
st.caption("Model Word2Vec digunakan untuk mengubah kata menjadi representasi vektor berdimensi rendah.")

# ====== Sample Table of Word Vectors ======
st.subheader("📌 Contoh Vektor Kata")
words = list(model.wv.index_to_key[:20])  
vectors = [model.wv[word] for word in words]

df_vectors = pd.DataFrame(vectors, index=words)
st.dataframe(df_vectors)

# ====== Visualisasi PCA ======
st.subheader("📊 Visualisasi Word Embedding (PCA)")

pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=reduced[:,0], y=reduced[:,1])

for i, word in enumerate(words):
    ax.text(reduced[i,0]+0.01, reduced[i,1]+0.01, word, fontsize=9)

st.pyplot(fig)

# input uji coba word2vec
st.subheader("🔍 Cari Kata Serupa")
input_word = st.text_input("Masukkan kata:", "")

if input_word:
    if input_word in model.wv:
        similar_words = model.wv.most_similar(input_word, topn=10)
        st.write("🔁 Kata serupa dengan:", input_word)
        st.table(pd.DataFrame(similar_words, columns=["Kata", "Kesamaan"]))
    else:
        st.warning(f"Kata '{input_word}' tidak ditemukan di model.")
