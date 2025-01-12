import openai
import numpy as np
import os
from dotenv import load_dotenv
import re
import requests
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from bs4 import BeautifulSoup

# ----------------------
# Load Environment Variables
# ----------------------
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")
WEBSITE_URL = os.getenv("WEBSITE_URL")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# ----------------------
# Functions
# ----------------------

# Function to get embeddings from OpenAI
def get_embeddings(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",  # Using OpenAI embedding model
            input=text
        )
        return np.array([embedding['embedding'] for embedding in response['data']])
    except Exception as e:
        return f"Error getting embeddings: {e}"

# Function to extract PDF text
def extract_pdf_text(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to scrape website content
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        return f"Error scraping website: {e}"

# Function to summarize content
def summarize_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize the following text for clarity and conciseness."},
                {"role": "user", "content": text}
            ],
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error summarizing text: {e}"

# Function to retrieve the most relevant content based on cosine similarity
def retrieve_relevant_content(user_question, content_embeddings, documents):
    question_embedding = get_embeddings(user_question)
    similarities = cosine_similarity(question_embedding, content_embeddings)
    best_match_index = np.argmax(similarities)
    return documents[best_match_index]

# Function to generate OpenAI response based on retrieved content
def chat_with_ai(user_question, website_text, pdf_text, content_embeddings, documents, chat_history):
    # Retrieve the most relevant content
    relevant_content = retrieve_relevant_content(user_question, content_embeddings, documents)

    # Combine the relevant content and user query for GPT model
    combined_context = f"""
    You have the following context to answer the question:

    Relevant Content: {relevant_content}

    User Question: {user_question}
    """
    
    messages = [
        {"role": "system", "content": "As an Aibytec chatbot, you are responsible for guiding the user through Aibytec’s services."}
    ]

    for entry in chat_history[-5:]:
        messages.append({"role": "user", "content": entry['user']})
        messages.append({"role": "assistant", "content": entry['bot']})

    messages.append({"role": "user", "content": f"{combined_context}"})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating response: {e}"

# Function to validate email
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# Function to validate name
def is_valid_name(name):
    return len(name) > 0

# Function to validate contact number
def is_valid_contact_no(contact_no):
    return len(contact_no) >= 10 and len(contact_no) <= 15 and contact_no.isdigit()

# Function to send email
def send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode):
    try:
        subject = "New Profile Submission from AIByTec"
        body = f"""
        Name: {name}
        Email: {email}
        Contact No.: {contact_no}
        
        Specific Needs and Challenges: {specific_needs_and_challenges}
        Training Preferred: {training}
        Mode of Training: {mode_of_training}
        Preferred Time/Mode of Contact: {prefered_time_contact_mode}
        """
        
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email via SMTP server
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        
        return "Email sent successfully!"
    except Exception as e:
        return f"Error sending email: {e}"

# ----------------------
# Streamlit UI and App Logic
# ----------------------

st.set_page_config(page_title="Student Profile & AI Chatbot", layout="wide")

# Session State Initialization
if "page" not in st.session_state:
    st.session_state['page'] = 'form'
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# ----------------------
# PAGE 1: User Info Form
# ----------------------
if st.session_state['page'] == 'form':
    st.subheader("Complete Your Profile")
    
    with st.form(key="user_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        contact_no = st.text_input("Contact No.")
        specific_needs_and_challenges = st.text_input("Task to be performed")
        training = st.text_input("Preferred course")
        mode_of_training = st.text_input("Online/Onsite")
        prefered_time_contact_mode = st.text_input("Preferred time/mode of contact")

        col1, col2 = st.columns([1, 1])
        with col1:
            submitted = st.form_submit_button("Complete Profile to Chat!")
        with col2:
            continue_chat = st.form_submit_button("Chat with AIByTec Bot")
        
        if submitted:
            if not is_valid_name(name):
                st.warning("Please enter a valid name.")
            elif not is_valid_email(email):
                st.warning("Please enter a valid email address.")
            elif not is_valid_contact_no(contact_no):
                st.warning("Please enter a valid contact number (10-15 digits).")
            elif not specific_needs_and_challenges or not training or not mode_of_training or not prefered_time_contact_mode:
                st.warning("Please fill out all fields.")
            else:
                email_status = send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode)
                st.success(email_status)
                st.session_state['page'] = 'chat'
                st.rerun()
        
        if continue_chat:
            st.session_state['page'] = 'chat'
            st.rerun()

# ----------------------
# PAGE 2: Chatbot Interface
# ----------------------
elif st.session_state['page'] == 'chat':
    # Initialize chat history with a greeting from the bot
    if not st.session_state['chat_history']:
        st.session_state['chat_history'].append({
            "user": "", 
            "bot": "Hello! I'm your AIByTec chatbot. How can I assist you today?"
        })
    
    # Display chat history
    for entry in st.session_state['chat_history']:
        if entry['user']:  # Show user messages
            st.markdown(f"<div>{entry['user']}</div>", unsafe_allow_html=True)
        if entry['bot']:  # Show bot messages
            st.markdown(f"<div>{entry['bot']}</div>", unsafe_allow_html=True)
    
    # Load PDF and Website content once
    pdf_text = extract_pdf_text(PDF_PATH) if os.path.exists(PDF_PATH) else "PDF file not found."
    website_text = scrape_website(WEBSITE_URL)

    # Combine the content and generate embeddings
    documents = [pdf_text, website_text]
    content_embeddings = get_embeddings([pdf_text, website_text])

    # Fixed input bar at bottom
    user_input = st.chat_input("Type your question here...", key="user_input_fixed")
    if user_input:
        # Display bot's response
        with st.spinner("Generating response..."):
            bot_response = chat_with_ai(user_input, website_text, pdf_text, content_embeddings, documents, st.session_state['chat_history'])
        # Append user query and bot response to chat history
        st.session_state['chat_history'].append({"user": user_input, "bot": bot_response})
        # Re-run to display updated chat history
        st.rerun()



























