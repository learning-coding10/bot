# import streamlit as st
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from PyPDF2 import PdfReader
# import requests
# from bs4 import BeautifulSoup
# import openai
# import os
# from dotenv import load_dotenv

# # ----------------------
# # Load Environment Variables
# # ----------------------
# load_dotenv()

# SENDER_EMAIL = os.getenv("SENDER_EMAIL")
# SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
# RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
# openai.api_key = os.getenv("OPENAI_API_KEY")
# PDF_PATH = os.getenv("PDF_PATH")
# WEBSITE_URL = os.getenv("WEBSITE_URL")

# # ----------------------
# # Functions and Rest of the Script
# # ----------------------

# # Function to send email
# def send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode):
#     subject = "New User Profile Submission"
#     body = f"""
#     New Student Profile Submitted:

#     Name: {name}
#     Email: {email}
#     Contact No.: {contact_no}
#     Task to be Performed: {specific_needs_and_challenges}
#     Preferred Course: {training}
#     Online/Onsite: {mode_of_training}
#     Preferred Time/Mode of Contact: {prefered_time_contact_mode}
#     """
#     message = MIMEMultipart()
#     message['From'] = SENDER_EMAIL
#     message['To'] = RECEIVER_EMAIL
#     message['Subject'] = subject
#     message.attach(MIMEText(body, 'plain'))
#     try:
#         server = smtplib.SMTP("smtp.gmail.com", 587)
#         server.starttls()
#         server.login(SENDER_EMAIL, SENDER_PASSWORD)
#         server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
#         server.quit()
#         st.success("Email sent successfully!")
#     except Exception as e:
#         st.error(f"Error sending email: {e}")

# # Function to extract PDF text
# def extract_pdf_text(file_path):
#     try:
#         reader = PdfReader(file_path)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() + "\n"
#         return text
#     except Exception as e:
#         st.error(f"Error reading PDF: {e}")
#         return ""

# # Function to scrape website content
# def scrape_website(url):
#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, "html.parser")
#         return soup.get_text()
#     except Exception as e:
#         return f"Error scraping website: {e}"

# # Function to generate OpenAI response
# def chat_with_ai(user_question, website_text, pdf_text, chat_history):
#     combined_context = f"Website Content:\n{website_text}\n\nPDF Content:\n{pdf_text}"
#     messages = [{"role": "system", "content": "You are a helpful assistant. Use the provided content."}]
#     for entry in chat_history:
#         messages.append({"role": "user", "content": entry['user']})
#         messages.append({"role": "assistant", "content": entry['bot']})
#     messages.append({"role": "user", "content": f"{combined_context}\n\nQuestion: {user_question}"})

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             stream=False
#         )
#         return response['choices'][0]['message']['content']
#     except Exception as e:
#         return f"Error generating response: {e}"

# # ----------------------
# # Streamlit UI and App Logic
# # ----------------------

# st.set_page_config(page_title="Student Profile & AI Chatbot", layout="wide")

# # Session State Initialization
# if "page" not in st.session_state:
#     st.session_state['page'] = 'form'
# if "chat_history" not in st.session_state:
#     st.session_state['chat_history'] = []

# # ----------------------
# # PAGE 1: User Info Form
# # ----------------------
# if st.session_state['page'] == 'form':
#     st.subheader("Complete Your Profile")
    
#     with st.form(key="user_form"):
#         name = st.text_input("Name")
#         email = st.text_input("Email")
#         contact_no = st.text_input("Contact No.")
#         specific_needs_and_challenges = st.text_input("Task to be performed")
#         training = st.text_input("Preferred course")
#         mode_of_training = st.text_input("Online/Onsite")
#         prefered_time_contact_mode = st.text_input("Preferred time/mode of contact")

#         # Create two columns for buttons
#         col1, col2 = st.columns([1, 1])  # equal width for both columns

#         with col1:
#             submitted = st.form_submit_button("Submit")
        
#         with col2:
#             continue_chat = st.form_submit_button("Skip")
        
#         if submitted:
#             if name and email and contact_no and specific_needs_and_challenges and training and mode_of_training and prefered_time_contact_mode:
#                 send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode)
#                 st.session_state['page'] = 'chat'
#                 st.rerun()
#             else:
#                 st.warning("Please fill out all fields.")
        
#         if continue_chat:
#             st.session_state['page'] = 'chat'
#             st.rerun()

# # ----------------------
# # PAGE 2: Chatbot Interface
# # ----------------------
# elif st.session_state['page'] == 'chat':
#     # Display chat history without headings
#     for entry in st.session_state['chat_history']:
#         # User Message
#         st.markdown(
#             f"""
#             <div style="
#                 background-color: #439DF6; 
#                 color: #fff;
#                 padding: 10px; 
#                 border-radius: 10px; 
#                 margin-bottom: 10px;
#                 width: fit-content;
#                 max-width: 80%;
#             ">
#                 {entry['user']}
#             </div>
#             """, 
#             unsafe_allow_html=True
#         )

#         # Assistant Message
#         st.markdown(
#             f"""
#             <div style="
#                 background-color:  #4a4a4a; 
#                 color: #fff;
#                 padding: 10px; 
#                 border-radius: 10px; 
#                 margin-bottom: 10px;
#                 margin-left: auto;
#                 width: fit-content;
#                 max-width: 80%;
#             ">
#                 {entry['bot']}
#             </div>
#             """, 
#             unsafe_allow_html=True
#         )

#     # Load PDF and Website content once
#     pdf_text = extract_pdf_text(PDF_PATH) if os.path.exists(PDF_PATH) else "PDF file not found."
#     website_text = scrape_website(WEBSITE_URL)

#     # Fixed input bar at bottom
#     user_input = st.chat_input("Type your question here...", key="user_input_fixed")

#     if user_input:
#         # Display bot's response
#         with st.spinner("Generating response..."):
#             bot_response = chat_with_ai(user_input, website_text, pdf_text, st.session_state['chat_history'])
        
#         # Append user query and bot response to chat history
#         st.session_state['chat_history'].append({"user": user_input, "bot": bot_response})
        
#         # Re-run to display updated chat history
#         st.rerun()


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

# Function to generate OpenAI response
def chat_with_ai(user_question, website_text, pdf_text, chat_history):
    combined_context = f"Website Content:\n{website_text}\n\nPDF Content:\n{pdf_text}"
    messages = [{"role": "system", "content": "You are a helpful assistant. Use the provided content."}]
    for entry in chat_history:
        messages.append({"role": "user", "content": entry['user']})
        messages.append({"role": "assistant", "content": entry['bot']})
    messages.append({"role": "user", "content": f"{combined_context}\n\nQuestion: {user_question}"})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=False
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
        mode_of_training = st.radio("Mode of Training", options=["Online", "Onsite"])
        preferred_time = st.text_input("Preferred Time")
        contact_mode = st.text_input("Contact Mode")

        col1, col2 = st.columns([1, 1])
        with col1:
            submitted = st.form_submit_button("Submit")
        with col2:
            continue_chat = st.form_submit_button("Skip")
        
        if submitted:
            if not is_valid_name(name):
                st.warning("Please enter a valid name.")
            elif not is_valid_email(email):
                st.warning("Please enter a valid email address.")
            elif not is_valid_contact_no(contact_no):
                st.warning("Please enter a valid contact number (10-15 digits).")
            elif not specific_needs_and_challenges or not training or not preferred_time or not contact_mode:
                st.warning("Please fill out all fields.")
            else:
                prefered_time_contact_mode = f"{preferred_time}, {contact_mode}"
                send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode)
                st.session_state['page'] = 'chat'
                st.rerun()
        
        if continue_chat:
            st.session_state['page'] = 'chat'
            st.rerun()

# # ----------------------
# # PAGE 2: Chatbot Interface
# # ----------------------
# elif st.session_state['page'] == 'chat':
#     for entry in st.session_state['chat_history']:
#         st.markdown(f"<div style='background-color: #439DF6; color: #fff; padding: 10px; border-radius: 10px; margin-bottom: 10px; width: fit-content; max-width: 80%;'>{entry['user']}</div>", unsafe_allow_html=True)
#         st.markdown(f"<div style='background-color: #4a4a4a; color: #fff; padding: 10px; border-radius: 10px; margin-bottom: 10px; margin-left: auto; width: fit-content; max-width: 80%;'>{entry['bot']}</div>", unsafe_allow_html=True)

#     pdf_text = extract_pdf_text(PDF_PATH) if os.path.exists(PDF_PATH) else "PDF file not found."
#     website_text = scrape_website(WEBSITE_URL)

#     user_input = st.chat_input("Type your question here...", key="user_input_fixed")

#     if user_input:
#         with st.spinner("Generating response..."):
#             bot_response = chat_with_ai(user_input, website_text, pdf_text, st.session_state['chat_history'])
#         st.session_state['chat_history'].append({"user": user_input, "bot": bot_response})
#         st.rerun()



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
                <div style="
                    background-color: #439DF6; 
                    padding: 10px;
                    color: #fff;
                    border-radius: 10px; 
                    margin-bottom: 10px;
                    width: fit-content;
                    max-width: 80%;
                    overflow: hidden;
                ">
                    {entry['user']}
                </div>
                """, 
                unsafe_allow_html=True
            )
        if entry['bot']:  # Show bot messages
            st.markdown(
                f"""
                <div style="
                    background-color: #4a4a4a; 
                    padding: 10px; 
                    color: #fff; 
                    border-radius: 10px; 
                    margin-bottom: 10px;
                    margin-left: auto;
                    width: fit-content;
                    max-width: 80%;
                    overflow: hidden;
                ">
                    {entry['bot']}
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    # Load PDF and Website content once
    pdf_text = extract_pdf_text(PDF_PATH) if os.path.exists(PDF_PATH) else "PDF file not found."
    website_text = scrape_website(WEBSITE_URL)

    # Fixed input bar at bottom
    user_input = st.chat_input("Type your question here...", key="user_input_fixed")
    if user_input:
        # Display bot's response
        with st.spinner("Generating response..."):
            bot_response = chat_with_ai(user_input, website_text, pdf_text, st.session_state['chat_history'])
        # Append user query and bot response to chat history
        st.session_state['chat_history'].append({"user": user_input, "bot": bot_response})
        # Re-run to display updated chat history
        st.rerun()


