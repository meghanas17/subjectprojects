import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import os

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
df = pd.read_csv("mini_multilingual.csv")
print("✅ Dataset loaded:", df.shape)
print(df['language'].value_counts())
import pandas as pd

df1 = pd.read_csv("mini_multilingual.csv")
df2 = pd.read_csv("extra_english.csv")
df = pd.concat([df1, df2], ignore_index=True)
df.to_csv("mini_multilingual_aug.csv", index=False, encoding='utf-8')

print("✅ Augmented dataset created:", df.shape)
print(df['language'].value_counts())


# -------------------------------
# 2️⃣ Encode Labels
# -------------------------------
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['language'])

# -------------------------------
# 3️⃣ Tokenize Characters
# -------------------------------
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['text'])

X = tokenizer.texts_to_sequences(df['text'])
maxlen = 40  # Limit to 40 characters
X = pad_sequences(X, maxlen=maxlen, padding='post')

y = df['label'].values

# -------------------------------
# 4️⃣ Split Data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5️⃣ Build Model (LSTM works better than RNN)
# -------------------------------
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=maxlen),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# 6️⃣ Train Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# -------------------------------
# 7️⃣ Evaluate
# -------------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# -------------------------------
# 8️⃣ Save Model & Tokenizer
# -------------------------------
model.save("language_model.h5")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n💾 Model, tokenizer, and label encoder saved successfully!")

# -------------------------------
# 9️⃣ Load & Test Function
# -------------------------------
def load_all():
    mdl = load_model("language_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return mdl, tok, le

def predict_language(text, mdl, tok, le):
    seq = tok.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = mdl.predict(seq)
    lang_idx = np.argmax(pred)
    lang_name = le.inverse_transform([lang_idx])[0]
    return lang_name

# -------------------------------
# 🔍 Interactive Test
# -------------------------------
print("\n🧠 Model ready for testing!")
model, tokenizer, label_encoder = load_all()
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("📊 F1 Score (weighted):", f1)
print("📌 Precision (weighted):", precision)
print("🔁 Recall (weighted):", recall)
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


while True:
    user_input = input("\nEnter text (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    lang = predict_language(user_input, model, tokenizer, label_encoder)
    print(f"→ Predicted Language: {lang}")


