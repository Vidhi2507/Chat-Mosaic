import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib import rcParams
rcParams['font.family'] = 'Segoe UI Emoji'

st.sidebar.title("Employee Communication & Productivity Tracking")
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
            st.title("Team Contribution")
            x,y = helper.user_engagement(df)
            col5,col6 = st.columns(2)
            with col5:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values)
                st.pyplot(fig)
            with col6:
                st.dataframe(y)

        #WORD CLOUD
        st.title("Frequent Work Keywords")
        df_wc = helper.create_WordCloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)


        #Words Analysis
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most Discussed Words')
        st.pyplot(fig)

        #emoji analysis
        st.title('Emoji Analysis for Tone Indication')
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
            st.header("Peak Collaboration Days")
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

        col9,col10 = st.columns(2)
        with col9:
            st.title("Sentiment Distribution")
            sentiment_counts = df["sentiment_label"].value_counts()
            st.write(sentiment_counts)
        with col10:
            st.title("Sentiment Bar Chart")
            st.bar_chart(df["sentiment_label"].value_counts())


        # MULTICLASS CLASSIFICATION - Emotion Analysis
        df["predicted_emotion"] = df["message"].apply(helper.predict_emotion)
        st.title("Emotion Analysis")
        st.dataframe(df[["message", "predicted_emotion"]])

        emotion_counts = df["predicted_emotion"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct="%0.1f%%")
        st.pyplot(fig)

        # AI-Generated Content Detection
        df["text_origin"] = df["message"].apply(helper.predict_ai_content)

        # Display results
        col_ai1, col_ai2 = st.columns(2)

        with col_ai1:
            st.subheader("Message Classification")
            st.dataframe(df[["user", "message", "text_origin"]].head(10))

        with col_ai2:
            st.subheader("Distribution")
            ai_counts = df["text_origin"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(ai_counts, labels=ai_counts.index, autopct="%0.1f%%", colors=['#66b3ff','#99ff99'])
            st.pyplot(fig)

        # Identification of 'Suspicious' users
        st.subheader("Most 'AI-Reliant' Users")
        ai_user_stats = df[df["text_origin"] == "AI-Generated"]["user"].value_counts()
        st.bar_chart(ai_user_stats)




        # Response Time
    st.title("Response Time Analysis")
    response_df = helper.create_response_time_dataset(df)
    st.dataframe(response_df.head())

    @st.cache_resource
    def train_model_cached(response_df):
            model, le = helper.train_response_time_model(response_df)
            return model, le

    model, le = train_model_cached(response_df)

    st.title("⏱ Response Time Prediction (ML)")

    with st.form("predict_form"):
        selected_responder = st.selectbox("Select Employee", response_df["responder"].unique())
        hour = st.slider("Hour of Day", 0, 23, 10)
        day = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        msg_length = st.slider("Message Length (characters)", 1, 200, 40)
        day_mapping = {"Mon":0, "Tue":1, "Wed":2, "Thu":3, "Fri":4, "Sat":5, "Sun":6}
        day_encoded = day_mapping[day]
        submitted = st.form_submit_button("Predict Response Time")

        if submitted:
            pred = helper.predict_response_time(selected_responder, hour, day_encoded, msg_length)
            st.success(f"📌 Predicted Response Time: {pred:.2f} minutes")
        