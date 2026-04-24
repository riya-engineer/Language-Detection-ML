import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("language.csv")

# Columns: Text, Language
X = df["Text"]
y = df["language"]

# Vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Model
model = MultinomialNB()
model.fit(X_vec, y)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained & saved!")