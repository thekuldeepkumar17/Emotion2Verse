import json
import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# Load Dataset
# -----------------------------

with open("dataset/gita_verses.json","r",encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# -----------------------------
# Text Preprocessing Function
# -----------------------------

stop_words = set(stopwords.words('english'))

def preprocess(text):

    text = text.lower()
    tokens = word_tokenize(text)

    words = [w for w in tokens if w.isalpha()]

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

df["processed"] = df["text"].apply(preprocess)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(df["processed"])

# -----------------------------
# Recommendation Function
# -----------------------------

def recommend_verse(user_input, top_n=3):

    processed_input = preprocess(user_input)

    user_vec = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vec, tfidf_matrix)

    similar_indices = similarity.argsort()[0][-top_n:][::-1]

    results = df.iloc[similar_indices]

    return results

# -----------------------------
# User Interaction
# -----------------------------

print("\nBhagavad Gita Emotion Based Verse Recommender\n")

user_input = input("Enter how you feel: ")

results = recommend_verse(user_input)

print("\nRecommended Verses:\n")

for i,row in results.iterrows():

    print(f"Chapter {row['chapter']} Verse {row['verse']}")
    print(row['text'])
    print()