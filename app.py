import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import webbrowser

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Say something!")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results; {e}"

# Function to perform basic NLP tasks
def analyze_text(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment = sentiment_analyzer.polarity_scores(text)
    return tokens, pos_tags, sentiment

# Streamlit frontend
st.title("Speech to Text Converter with NLP")

if st.button("Start Recording"):
    result = speech_to_text()
    st.write("You said:")
    st.success(result)
    
    if st.button("Search on Google"):
        webbrowser.open(f"https://www.google.com/search?q={result}")

    # Perform NLP tasks
    tokens, pos_tags, sentiment = analyze_text(result)
    
    st.write("### Tokens")
    st.write(tokens)

    st.write("### Part of Speech Tags")
    st.write(pos_tags)

    st.write("### Sentiment Analysis")
    st.write(sentiment)
