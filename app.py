import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# Streamlit Page Config (FIRST LINE)
# -----------------------------------
st.set_page_config(page_title="Multilingual Language Detector", layout="centered")

# -----------------------------------
# Load Model, Tokenizer, LabelEncoder
# -----------------------------------
@st.cache_resource
def load_all():
    model = load_model("language_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, tok, le


model, tokenizer, label_encoder = load_all()
maxlen = 40

# -----------------------------------
# Streamlit UI
# -----------------------------------

st.title("🌍 Multilingual Language Detector")
st.write("Enter any text and the model will detect the language.")

# Input box
user_input = st.text_input("Enter some text:", "")

def predict_language(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(seq, verbose=0)
    lang_idx = np.argmax(pred)
    return label_encoder.inverse_transform([lang_idx])[0]

# Predict button
if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter text!")
    else:
        lang = predict_language(user_input)
        st.success(f"Predicted Language: **{lang}**")
