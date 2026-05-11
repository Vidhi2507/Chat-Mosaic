import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset you provided
df = pd.read_csv('dataset\AIvsHuman.csv')

# 2. Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Clean text and Map labels: 'ai' -> 1, 'human' -> 0
df["clean_text"] = df["text"].apply(preprocess_text)
df["label"] = df["human_or_ai"].map({'ai': 1, 'human': 0})

df = df.dropna(subset=['label', 'clean_text'])
df['label'] = df['label'].astype(int)

# 3. Split data
X = df["clean_text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorization (Turning text into numbers)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Decision Tree Model
# Using a Decision Tree as requested
model = DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=42)
model.fit(X_train_vec, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 7. Save the model and vectorizer for the app to use
joblib.dump(model, "ai_detector_model.pkl")
joblib.dump(vectorizer, "ai_detector_vectorizer.pkl")
print("Model and Vectorizer saved successfully!")