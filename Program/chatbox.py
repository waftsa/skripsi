import nltk

# Unduh resource 'punkt'
nltk.download('punkt')

from nltk.tokenize import word_tokenize
import re
from collections import defaultdict

# Fungsi untuk normalisasi teks Arab
def normalize_arabic(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ؤ', 'ء', text)
    return text

# Fungsi untuk tokenisasi teks
def tokenize_text(text):
    return word_tokenize(text.lower())

# Fungsi untuk membuat indeks kata kunci
def create_index(data):
    index = defaultdict(list)
    for entry in data:
        tokens = tokenize_text(entry['text_id'])
        for token in tokens:
            index[token].append({
                "surah": entry["surah"],
                "ayah": entry["ayah"],
                "text_ar": entry["text_ar"],
                "text_id": entry["text_id"]
            })
    return index

# Contoh data
data = [
    {
        "surah": 1,
        "ayah": 1,
        "text_ar": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "text_id": "Dengan nama Allah Yang Maha Pengasih, Maha Penyayang."
    },
    {
        "surah": 1,
        "ayah": 2,
        "text_ar": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        "text_id": "Segala puji bagi Allah, Tuhan seluruh alam."
    },
    # Tambahkan data lainnya
]

# Membuat indeks kata kunci
index = create_index(data)

# Fungsi untuk pencarian kata kunci
def search(query, index):
    tokens = tokenize_text(query)
    results = []
    for token in tokens:
        if token in index:
            results.extend(index[token])
    return results

# Contoh pencarian
query = "Berikan ayat tentang memuji tuhan"
results = search(query, index)

for result in results:
    print(f"Surah: {result['surah']}, Ayah: {result['ayah']}")
    print(f"Teks Arab: {result['text_ar']}")
    print(f"Terjemahan: {result['text_id']}")
    print()
