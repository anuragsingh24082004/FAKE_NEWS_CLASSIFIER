import pandas as pd
import numpy as np
import re
import string
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
import nltk

nltk.download('stopwords')
tqdm.pandas()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
punct_table = str.maketrans('', '', string.punctuation)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.translate(punct_table)
    text = re.sub(r'\d+', '', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

print("✅ Reading data...")
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

print("✅ Merging and shuffling data...")
df = pd.concat([true_df, fake_df], axis=0).sample(frac=1).reset_index(drop=True)

# Optional: For testing on smaller sample
# df = df.sample(n=3000, random_state=42)

print("✅ Preprocessing text...")
df["text"] = df["title"] + " " + df["text"]
df["text"] = df["text"].progress_apply(clean_text)

X = df["text"]
y = df["label"]

print("✅ TF-IDF vectorization...")
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
X = tfidf.fit_transform(X)

print("✅ Splitting train/test data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Training model...")
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)

print("✅ Evaluating model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("✅ Saving model and vectorizer...")
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ All done!")