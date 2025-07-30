import streamlit as st
from pathlib import Path
from PIL import Image

st.title('ANALISIS SENTIMEN TERHADAP ISU GENOSIDA ISRAEL KEPADA PALESTINA')

st.write("""
        Aplikasi ini merupakan hasil dari penelitian skripsi penulis yang berjudul *Analisis Sentimen terhadap Isu Genosida Israel kepada Palestina pada Media Sosial X (Twitter)*.

        Aplikasi ini berisi proses dari penelitian ini, bermula dari scraping dataset, pre-processing, labelling, balancing, word2vec. kemudian adapun komparasi model paling efektif yang digunakan dalam penelitian ini. dan pengguna pada mencoba mengklasifikasi sentimen secara real time. 
        """) 
st.markdown("---")

st.subheader("👨‍💻 Tim Pengembang")

BASE_DIR = Path(__file__).resolve().parent
image_developer = BASE_DIR / "image" / "default.jpg"
image_dosbim = BASE_DIR / "image" / "default.jpg"
image_dosbim2 = BASE_DIR / "image" / "default.jpg"


col1, col2, col3 = st.columns(3)

with col1:
    if image_developer.exists():
        st.image(str(image_developer), width=200)
        st.write("*Wafa Tsabita*")
        st.write("**Developer**")
    else:
        st.error(f"Gambar tidak ditemukan: {image_developer}")

with col2:
    if image_dosbim.exists():
        st.image(str(image_dosbim), width=200)
        st.write("*Dr. Afrida Helen, S.T., M.Kom.*")
        st.write("**Pembimbing Utama**")
    else:
        st.error(f"Gambar tidak ditemukan: {image_dosbim}")

with col3:
    if image_dosbim2.exists():
        st.image(str(image_dosbim2), width=200)
        st.write("*Mira Suryani, S.Pd, M.Kom.*")
        st.write("**Co-Pembimbing**")
    else:
        st.error(f"Gambar tidak ditemukan: {image_dosbim2}")

st.markdown("---")

st.subheader("🖋️ Tim Annotator")
st.write("""
        - Annotator 1 : Musfirah Qisthi Tardauna 
        - Annotator 2 : Fatimah Noor Albirkah  
         """)