import pandas as pd
import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sqlite3

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
            return text  # Return only the recognized text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results; {e}"

# Function to perform NLP tasks
def perform_nlp_tasks(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return tokens, pos_tags, sentiment

# Function to search employees in SQLite database based on search parameters
def search_employees(sqlite_db_path, search_params):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(sqlite_db_path)

        # Build SQL query dynamically based on search parameters
        query = "SELECT * FROM employees WHERE 1=1"
        params = []
        
        if 'employee_name' in search_params and search_params['employee_name']:
            query += " AND [Full Name] LIKE ?"
            params.append(f"%{search_params['employee_name']}%")
        
        if 'department' in search_params and search_params['department']:
            query += " AND Department LIKE ?"
            params.append(f"%{search_params['department']}%")
        
        if 'job_title' in search_params and search_params['job_title']:
            query += " AND [Job Title] LIKE ?"
            params.append(f"%{search_params['job_title']}%")
        
        # Execute SQL query
        st.write(f"Executing SQL query: {query}")
        st.write(f"With parameters: {params}")  
        if params:
            df = pd.read_sql(query, conn, params=params)
        else:
            df = pd.read_sql(query, conn)

        # Close connection
        conn.close()

        return df

    except Exception as e:
        return f"Error occurred: {e}"
    

# Streamlit frontend
st.title("Employee Data Search")

# Placeholder for speech recognition result
result = ""

# Search section
st.header("Search")

# Option to use speech recognition or manual input
option = st.radio("Select search method:", ("Speech Recognition", "Manual Input"))

# Define search parameters
search_params = {}

if option == "Speech Recognition":
    st.info("Say the search parameter followed by the value. For example, 'Employee Name Robert Patel', 'Department Sales', or 'Job Title Analyst'.")
    if st.button("Start Recording", key="start_recording_btn"):
        result = speech_to_text()
        st.write("You said:")
        st.success(result)
        
        # Perform NLP tasks
        tokens, pos_tags, sentiment = perform_nlp_tasks(result)
        
        # Process the recognized text to extract search parameters
        if "employee name" in result.lower():
            search_params['employee_name'] = result.split("employee name")[-1].strip()
        elif "department" in result.lower():
            search_params['department'] = result.split("department")[-1].strip()
        elif "job title" in result.lower():
            search_params['job_title'] = result.split("job title")[-1].strip()
        else:
            st.error("Please say a valid search parameter followed by the value.")
        
        st.write(f"Processed Search Parameters: {search_params}")

        # Directly perform the search after processing the speech input
        sqlite_db_path = '/Users/rutvijjadhav/Downloads/employee_data.db'  # Update with your database path
        search_result = search_employees(sqlite_db_path, search_params)
        
        # Display results in split columns
        col1, col2 = st.columns(2)
        with col1:
            st.write("### NLP Tasks Results")
            st.write("Tokens:", tokens)
            st.write("POS Tags:", pos_tags)
            st.write("Sentiment Analysis:", sentiment)
        
        # Add a Streamlit divider for separation
        st.divider()
        
        with col2:
            if isinstance(search_result, pd.DataFrame):
                st.write("### Search Results")
                st.write(search_result)  # Display all columns from the DataFrame
            else:
                st.error(f"Failed to perform search: {search_result}")

elif option == "Manual Input":
    search_params['employee_name'] = st.text_input("Employee Name:")
    search_params['department'] = st.text_input("Department:")
    search_params['job_title'] = st.text_input("Job Title:")

    # SQLite database path
    sqlite_db_path = '/Users/rutvijjadhav/Downloads/employee_data.db'  # Update with your database path

    # Search button for manual input
    if st.button("Search", key="search_btn"):
        st.write(f"Final Search Parameters: {search_params}")
        search_result = search_employees(sqlite_db_path, search_params)
        
        # Display results in split columns
        col1, col2 = st.columns(2)
        with col1:
            st.write("### NLP Tasks Results")
            st.write("Tokens: N/A (Manual input selected)")
            st.write("POS Tags: N/A (Manual input selected)")
            st.write("Sentiment Analysis: N/A (Manual input selected)")
        
        # Add a Streamlit divider for separation
        st.divider()
        
        with col2:
            if isinstance(search_result, pd.DataFrame):
                st.write("### Search Results")
                st.write(search_result)  # Display all columns from the DataFrame
            else:
                st.error(f"Failed to perform search: {search_result}")
