import pandas as pd

df = pd.read_csv('D:\\Vidhi\\DM Project\\WhatsappChatAnalyzer\\dataset\\twitter emotions.csv')
# print(df.head())  

import re

def preprocess_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)      # remove urls
    text = re.sub(r"@\w+", "", text)                # remove mentions like @ArcticFantasy
    text = re.sub(r"[^a-zA-Z\s]", " ", text)        # remove symbols
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    df["clean_text"] = df["text"].apply(preprocess_text)
    # print(df["emotion"].value_counts())
    df["emotion"] = df["emotion"].replace("NONE", "neutral")

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    X = df["clean_text"]
    y = df["emotion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1,2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "emotion_model.pkl")
    joblib.dump(vectorizer, "emotion_vectorizer.pkl")

