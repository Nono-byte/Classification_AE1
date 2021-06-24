"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import nltk
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
# Data dependencies
import pandas as pd
from sklearn.metrics import classification_report

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
train_df = raw.copy()

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    
    st.markdown(""" <style>div.stButton > button:first-child {
    background-color: #0000FF;color:white;font-size:20px;height:3em;width:30em;border-radius:10px 10px 10px 10px;
    </style>
    """, unsafe_allow_html=True)
    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")
    st.image("https://i.gifer.com/RD07.gif")
    st.header("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information","Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Social Media data is a huge source of Big Data in the form of chats, messages, news feeds, etc. Text analytics is a process that helps analyse this unstructured data and look for useful trends."+
        "Twitter data is a powerful source of information on a wide list of topics. This data can be analyzed to find trends related to specific topics, measure popular sentiment, obtain feedback on past decisions and also help make future decisions."+
        "This app is a tool that is designed to analyse customer sentiment on climate change based on their tweets, and thus help in the business decision making process.")


        st.image('wordcloud2.png')
        
        st.image('sentiment.png')
        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page

    # Building out the predication page
    if selection == "Prediction":
        st.info("Climate change prediction tool ")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")

        if st.button("Predict"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/classifier.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.write("Text Categorized as: {}".format(prediction))
            sent=""
            if(prediction[0]==0):
                st.success("Neutral sentiment")
                sent="Neutral sentiment"
            elif(prediction[0]==-1):
                st.success("Negative sentiment")
                sent="Negative sentiment"
            elif(prediction[0]==1):
                st.success("Positive sentiment")
                sent="Positive sentiment"
            elif(prediction[0]==2):
                st.success("News sentiment")
                sent="News sentiment"
                
            st.success("The text: "+tweet_text+" is classified as a "+sent+" according to our models prediction")
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
