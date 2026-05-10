from turtle import pd

import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()   #getting the byte file

    data = bytes_data.decode("utf-8")  #converting the uploaded file in txt format
    #st.text(data) printing the data on the screen
    df = preprocessor.preprocess(data)
    st.dataframe(df)


    #getting unique users for dropdown
    user_list = df['user'].unique().tolist()
    user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt User", user_list)

    #STATS AREA
    num_messages,words,media,links = helper.fetch_stats(selected_user,df)

    #conditional rendering after click of Show Analysis Button
    if st.sidebar.button("Show Analysis"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Links shared")
            st.title(links)
        with col4:
            st.header("Media shared")
            st.title(media)

    #USER ENGAGEMENT
        if selected_user == 'Overall':
            st.title("User Engagement")
            x,y = helper.user_engagement(df)
            col5,col6 = st.columns(2)
            with col5:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values)
                st.pyplot(fig)
            with col6:
                st.dataframe(y)

        #WORD CLOUD
        st.title("Word Cloud")
        df_wc = helper.create_WordCloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)


        #Words Analysis
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        #emoji analysis
        st.title('Emoji Analysis')
        emoji_df = helper.emoji_analysis(selected_user,df)

        col7,col8 = st.columns(2)
        with col7:
            st.title('Dataframe')
            st.dataframe(emoji_df)
        with col8:
            st.title('Pie Chart')
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        st.title("Sentiment Analysis")
        df[["cleaned_text", "sentiment_score", "sentiment_label"]] = df["message"].apply(lambda x: pd.Series(helper.get_sentiment(x)))
        #for each person
        st.dataframe(df)

        sentiment_counts = df["sentiment_label"].value_counts()
        st.write(sentiment_counts)
        st.bar_chart(df["sentiment_label"].value_counts())


        # MULTICLASS CLASSIFICATION - Emotion Analysis
        df["predicted_emotion"] = df["message"].apply(helper.predict_emotion)
        st.title("Emotion Analysis")
        st.dataframe(df[["message", "predicted_emotion"]])

        emotion_counts = df["predicted_emotion"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct="%0.1f%%")
        st.pyplot(fig)

        emotion_weights = {
        "joy": 1,"happy": 1,
            "neutral": 0,
            "sadness": 2,
            "sad": 2,
            "fear": 2,
            "anger": 3,
            "disgust": 3,
            "surprise": 2
        }
        df["emotion_score"] = df["predicted_emotion"].map(emotion_weights).fillna(1)
        most_emotional_user = df.groupby("user")["emotion_score"].sum().sort_values(ascending=False)
        st.title("Most Emotional User")
        st.dataframe(most_emotional_user[0:1])

        # #TOPIC MODELING
        # st.title("Topic Modeling")
        # df,topics = helper.topic_modelling(df)
        # st.dataframe(df)
        # topic_counts = df["topic"].value_counts().sort_index()

        # for topic_num, count in topic_counts.items():
        #     st.subheader(f"Topic {topic_num+1} (Messages: {count})")
        #     words = topics[topic_num][1]
        #     st.write(" ".join(words))

        # Summary of chat
        st.title("Chat Summary")
        chat_text = " ".join(df["message"].astype(str).tolist())

        summary = helper.summarize_text(chat_text)
        st.write(summary)