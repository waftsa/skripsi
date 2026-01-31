import streamlit as st
from pathlib import Path


st.header('ANALISIS KOMPARASI MODEL MACHINE LEARNING DAN DEEP LEARNING')

st.write("""
Aplikasi ini merupakan hasil dari penelitian skripsi berjudul 
*Analisis Komparasi Model Machine Learning dan Deep Learning untuk Klasifikasi Sentimen Isu Genosida Palestina pada Media Sosial X*.

Aplikasi ini menampilkan tahapan penelitian yang dilakukan, mulai dari proses pengumpulan data (scraping tweet), 
text preprocessing, pelabelan data, data balancing, hingga representasi teks menggunakan Word2Vec dan TF-IDF. 
Selain itu, aplikasi ini menyajikan hasil komparasi kinerja beberapa model klasifikasi sentimen, yaitu 
Naive Bayes, Support Vector Machine (SVM), Long Short-Term Memory (LSTM), dan Bidirectional LSTM (BiLSTM).
""")

st.markdown("---")


st.subheader("👨‍💻 Tim Pengembang")

col1, col2, col3 = st.columns(3)

with col1:
        st.write("**Wafa Tsabita**")
        st.write("*Developer*")

with col2:
        st.write("**Dr. Afrida Helen, S.T., M.Kom.**")
        st.write("*Pembimbing Utama*")

with col3:
        st.write("**Dr. Mira Suryani, S.Pd, M.Kom.**")
        st.write("*Co-Pembimbing*")


st.markdown("---")

st.subheader("🖋️ Tim Annotator")
st.write("""
        - Annotator 1 : Musfirah Qisthi Tardauna 
        - Annotator 2 : Fatimah Noor Albirkah  
         """)