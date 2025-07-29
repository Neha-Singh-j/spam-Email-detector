import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load and preprocess data - FIXED READING OF CSV
df = pd.read_csv("spam.csv", sep=",", header=None, skiprows=1, names=["label", "message"])

# Check data
print("First 5 rows:")
print(df.head())

# Check for invalid labels
print("\nUnique labels:", df["label"].unique())

# Map labels to numbers
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Check for NaN in labels
print("\nNaN values in labels:", df["label_num"].isna().sum())

# Drop rows with NaN labels (if any)
df = df.dropna(subset=["label_num"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.2, random_state=42
)

# Fill NaN in text data
X_train = X_train.fillna("")
X_test = X_test.fillna("")

# Vectorize and train
vectorizer = TfidfVectorizer()
X_train_tf = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_tf, y_train)

# Evaluate
X_test_tf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tf)
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))