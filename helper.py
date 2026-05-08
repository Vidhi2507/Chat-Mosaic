from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from SentimentAnalysismodel import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer


# importing the trained model
import joblib
trained_model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

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
    score = trained_model.predict(text_vec)[0]

    if score > 0.2:
        label = "Positive"
    elif score < -0.2:
        label = "Negative"
    else:
        label = "Neutral"

    return score, label


#Clustering model for user clustering based on their activity patterns
def cluster_users(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    features = df[['only_date','year','month_num','hour','minute']]  
    
    features['only_date'] = features['only_date'].apply(lambda x: x.toordinal())


    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3)  # You can choose the number of clusters based on your data
    df['cluster'] = kmeans.fit_predict(features)
    return df[['user', 'cluster']]


