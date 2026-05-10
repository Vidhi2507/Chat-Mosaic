from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from SentimentAnalysismodel import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# importing the trained model
import joblib
trained_model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

emotion_model = joblib.load("emotion_model.pkl")
emotion_vectorizer = joblib.load("emotion_vectorizer.pkl")

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

    if clean_text == "" or len(clean_text.split()) <= 2:
        return "neutral"

    vec = emotion_vectorizer.transform([clean_text])
    emotion = emotion_model.predict(vec)[0]

    return emotion
    

def topic_modelling(df):
    import re
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    def clean_topic_text(text):
        text = str(text).lower()
        # remove urls
        text = re.sub(r"http\S+|www\S+", "", text)
        # remove mentions and numbers
        text = re.sub(r"@\d+", "", text)
        text = re.sub(r"\d+", "", text)
        # remove punctuation
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        # remove stopwords
        words = [w for w in text.split() if w not in stop_words and len(w) > 2]
        return " ".join(words)
    df["clean_message"] = df["message"].apply(clean_topic_text)

    # remove empty messages
    df = df[df["clean_message"].str.strip() != ""]
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(df["clean_message"])
    from sklearn.decomposition import LatentDirichletAllocation

    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(X)

    topic_distribution = lda_model.transform(X)
    df["topic"] = topic_distribution.argmax(axis=1)
    topics = {}
    for topic_num, comp in enumerate(lda_model.components_):
        word_indices = comp.argsort()[-10:][::-1]
        words = [vectorizer.get_feature_names_out()[i] for i in word_indices]
        topics[topic_num] = words
    return df,topics



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(inputs["input_ids"], max_length=80, min_length=30)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)