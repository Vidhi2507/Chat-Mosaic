import joblib
import pandas as pd

import re

def preprocess_text(text):
    # lowercasing
    text = text.lower()
    
    # removing urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # remove punctuation but keep letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # removing numbers
    text = re.sub(r'\d+', '', text)

    # removing extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
    

#Reading a textfile with label as filename and text as the content of the file on each line
with open('dataset/negative.txt', 'r') as f:
    negative_texts = [line.strip() for line in f]

with open('dataset/positive.txt', 'r') as f:
    positive_texts = [line.strip() for line in f]

# Create a DataFrame with the texts and their labels
df = pd.DataFrame({
    'text': negative_texts + positive_texts,
    'sentiment_score': [0] * len(negative_texts) + [1] * len(positive_texts)
})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df['text'] = df['text'].apply(preprocess_text)

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1,2),
    sublinear_tf=True,
    min_df=2
)
X_tfidf = vectorizer.fit_transform(df['text'])


X = X_tfidf
y = df['sentiment_score']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# # hyperparameter tuning using GridSearchCV
# from sklearn.model_selection import GridSearchCV
# param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# best_alpha = grid_search.best_params_['alpha']
# print("Best alpha:", best_alpha)

# Predictions
y_pred = model.predict(X_test)

# print(model.predict(X_test[1]))
# print(y_test.iloc[1])

# classification Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))



# After training your_model
joblib.dump(model, 'trained_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')