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
    
    # removing extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

df = pd.read_csv('hinglish_english_dataset.csv')
df = df[['text', 'sentiment_score']]
# df['sentiment_score'] = df['sentiment_score'].apply(lambda x: 1 if x > 0 else 0)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR

model = LinearSVR()
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

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)



# After training your_model
joblib.dump(model, 'trained_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')