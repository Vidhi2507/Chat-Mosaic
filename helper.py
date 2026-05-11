from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from Models.SentimentAnalysismodel import preprocess_text
from sklearn.preprocessing import LabelEncoder


# importing the trained model
import joblib
trained_model = joblib.load('Trained_Models/trained_model.pkl')
vectorizer = joblib.load('Trained_Models/vectorizer.pkl')

emotion_model = joblib.load("Trained_Models/emotion_model.pkl")
emotion_vectorizer = joblib.load("Trained_Models/emotion_vectorizer.pkl")

ai_model = joblib.load("Trained_Models/ai_detector_model.pkl")
ai_detector_vectorizer = joblib.load("Trained_Models/ai_detector_vectorizer.pkl")

def fetch_stats(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user] #Masking the dataframe if user is specific

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    extract = URLExtract()
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)


def user_engagement(df):
    df = df[df['user']!='group_notification'] 
    x = df['user'].value_counts().head()
    y = round((df['user'].value_counts()/df.shape[0])*100,2)
    #converting it into dataframe and setting name of columns
    df = y.reset_index().rename(columns={'index':'name','user':'name','count':'percent'})
    return x,df



def create_WordCloud(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user] #Masking the dataframe if user is specific
    df = df[df['message'] != '<Media omitted>\n']
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user,df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_analysis(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(emojis)))
    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def get_sentiment(text):
    text = preprocess_text(text)
    text_vec = vectorizer.transform([text])
    score = trained_model.predict_proba(text_vec)[0][1]  # Probability of being positive

    if score >= 0.6:
        label = "Positive"
    elif score <= 0.4:
        label = "Negative"
    else:
        label = "Neutral"

    return text,score, label



def predict_emotion(text):
    clean_text = preprocess_text(text)

    if clean_text == "" or len(clean_text.split()) < 6:
        return "neutral"

    vec = emotion_vectorizer.transform([clean_text])
    emotion = emotion_model.predict(vec)[0]

    return emotion
    


def predict_ai_content(text):
    clean_text = preprocess_text(text)
    
    if clean_text == "" or len(clean_text.split()) < 10:
        return "Human" # Short messages are hard to classify as AI
        
    vec = ai_detector_vectorizer.transform([clean_text])
    prediction = ai_model.predict(vec)[0]
    
    return "AI-Generated" if prediction == 1 else "Human-Written"




def create_response_time_dataset(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    response_rows = []

    for i in range(len(df)-1):
        user = df.loc[i, "user"]
        msg_time = df.loc[i, "date"]

        # find next message from another user
        for j in range(i+1, len(df)):
            if df.loc[j, "user"] != user:
                next_time = df.loc[j, "date"]
                responder = df.loc[j, "user"]

                response_time_minutes = (next_time - msg_time).total_seconds() / 60

                response_rows.append({
                    "sender": user,
                    "responder": responder,
                    "message": df.loc[i, "message"],
                    "response_time": response_time_minutes,
                    "hour": msg_time.hour,
                    "dayofweek": msg_time.dayofweek,
                    "msg_length": len(str(df.loc[i, "message"]))
                })
                break

    response_df = pd.DataFrame(response_rows)

    # remove extreme outliers (like 2 days reply)
    response_df = response_df[response_df["response_time"] <= 720]  # 12 hours max
    response_df = response_df[response_df["response_time"] >= 0]

    return response_df

def train_response_time_model(response_df):
    response_df = response_df.copy()

    le = LabelEncoder()
    response_df["responder_encoded"] = le.fit_transform(response_df["responder"])

    X = response_df[["responder_encoded", "hour", "dayofweek", "msg_length"]]
    y = response_df["response_time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    joblib.dump(model, "response_time_rf_model.pkl")
    joblib.dump(le, "responder_encoder.pkl")

    return model, le



def predict_response_time(responder_name, hour, dayofweek, msg_length):
    rf_model = joblib.load("response_time_rf_model.pkl")
    le = joblib.load("responder_encoder.pkl")

    responder_encoded = le.transform([responder_name])[0]

    X_new = pd.DataFrame([[responder_encoded, hour, dayofweek, msg_length]],
                         columns=["responder_encoded", "hour", "dayofweek", "msg_length"])

    predicted_time = rf_model.predict(X_new)[0]
    return predicted_time


# def calculate_response_kpis(response_df):
#     avg_time = response_df["response_time_min"].mean()
#     median_time = response_df["response_time_min"].median()

#     fastest_user = response_df.groupby("responder")["response_time_min"].mean().idxmin()
#     slowest_user = response_df.groupby("responder")["response_time_min"].mean().idxmax()

#     fast_pct = (response_df["response_time_min"] <= 5).mean() * 100
#     sla_pct = (response_df["response_time_min"] <= 10).mean() * 100

#     return avg_time, median_time, fastest_user, slowest_user, fast_pct, sla_pct
    