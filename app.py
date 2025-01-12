import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import openai
import os
from dotenv import load_dotenv
import re  # For validation
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------
# Load Environment Variables
# ----------------------
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
openai.api_key = os.getenv("OPENAI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")
WEBSITE_URL = os.getenv("WEBSITE_URL")

# ----------------------
# Functions
# ----------------------

# Validate name, email, and contact number
def is_valid_name(name):
    return len(name.strip()) > 0

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_contact_no(contact_no):
    return re.match(r"^\+?\d{10,15}$", contact_no)

# Function to send email
def send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode):
    subject = "New User Profile Submission"
    body = f"""
    New Student Profile Submitted:

    Name: {name}
    Email: {email}
    Contact No.: {contact_no}
    Task to be Performed: {specific_needs_and_challenges}
    Preferred Course: {training}
    Mode of Training: {mode_of_training}
    Preferred Time/Mode of Contact: {prefered_time_contact_mode}
    """
    message = MIMEMultipart()
    message['From'] = SENDER_EMAIL
    message['To'] = RECEIVER_EMAIL
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
        server.quit()
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# Function to extract PDF text
def extract_pdf_text(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Function to scrape website content
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        return f"Error scraping website: {e}"

# Function to generate embeddings
def generate_embeddings(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",  # OpenAI's embedding model
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        return f"Error generating embeddings: {e}"

# Function to calculate cosine similarity
def calculate_cosine_similarity(query_embedding, document_embeddings):
    return cosine_similarity([query_embedding], document_embeddings)

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

# Function to generate OpenAI response using RAG-like approach
def chat_with_ai(user_question, website_text, pdf_text, chat_history):
    # Generate embeddings for website and PDF content
    pdf_embeddings = generate_embeddings(pdf_text)
    website_embeddings = generate_embeddings(website_text)

    # Generate embedding for user query
    query_embedding = generate_embeddings(user_question)

    # Calculate cosine similarity between query and document embeddings
    pdf_similarity = calculate_cosine_similarity(query_embedding, [pdf_embeddings])[0][0]
    website_similarity = calculate_cosine_similarity(query_embedding, [website_embeddings])[0][0]

    # Select the most relevant document (PDF or Website) based on similarity score
    if pdf_similarity > website_similarity:
        most_relevant_content = pdf_text
    else:
        most_relevant_content = website_text

    # Summarize the most relevant content
    summarized_relevant_content = summarize_text(most_relevant_content)

    # Combine context and user question for final GPT input
    combined_context = f"""
    The most relevant information is as follows:
    {summarized_relevant_content}

    Answer the user's question based on this context:
    """

    messages = [
        {"role": "system", "content": "You are an AI assistant providing answers based on relevant information."}
    ]

    # Add previous chat history
    for entry in chat_history[-5:]:
        messages.append({"role": "user", "content": entry['user']})
        messages.append({"role": "assistant", "content": entry['bot']})

    messages.append({"role": "user", "content": f"{combined_context}\n\nQuestion: {user_question}"})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating response: {e}"

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
                send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode)
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
            st.markdown(
                f"""
                <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <div style='color: #439DF6;'>👤</div>
                        <div style='max-width: 70%; 
                                    background-color: #439DF6; color: #fff; 
                                    padding: 10px; border-radius: 10px;'>
                            {entry['user']}
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        if entry['bot']:  # Show bot messages
            st.markdown(
                f"""
                <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <div style='color: #4a4a4a;'>🤖</div>
                        <div style='max-width: 70%; 
                                    background-color: #4a4a4a; color: #fff; 
                                    padding: 10px; border-radius: 10px;'>
                            {entry['bot']}
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

    user_question = st.text_input("Ask me anything:", "")
    if user_question:
        # Extract and process PDF/Website content
        website_text = scrape_website(WEBSITE_URL)
        pdf_text = extract_pdf_text(PDF_PATH)
        
        # Get AI response with RAG functionality
        ai_answer = chat_with_ai(user_question, website_text, pdf_text, st.session_state['chat_history'])
        
        # Add to chat history
        st.session_state['chat_history'].append({"user": user_question, "bot": ai_answer})
        st.session_state['chat_history'] = st.session_state['chat_history'][-5:]  # Keep the last 5 messages

        st.experimental_rerun()
