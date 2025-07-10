from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences

# Init app
app = Flask(__name__)
 
# Load model dan tokenizer
model = load_model('model/lstm_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Parameter input
MAX_SEQUENCE_LENGTH = 100  # sesuaikan dengan padding saat training

def predict_sentiment(text):
    # Preprocessing: tokenizing + padding
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Prediksi
    prediction = model.predict(padded)[0][0]

    # Threshold bisa kamu ubah, ini binary
    if prediction >= 0.6:
        return 'Positif'
    elif prediction <= 0.4:
        return 'Negatif'
    else:
        return 'Netral'

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        user_text = request.form['input_text']
        sentiment = predict_sentiment(user_text)
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
