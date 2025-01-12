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






















''' 
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

# Load Environment Variables
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
openai.api_key = os.getenv("OPENAI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")
WEBSITE_URL = os.getenv("WEBSITE_URL")

# Functions

# Function to send email
def send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode):
    subject = "New User Profile Submission"
    body = f"""
    New Student Profile Submitted:
    Name: {name}
    Email: {email}
    Contact No.: {contact_no}
    Specific Needs & Challenges: {specific_needs_and_challenges}
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
#             max_tokens=256,
#             temperature=0.5,
#             stream=False
#         )
#         return response['choices'][0]['message']['content']
#     except Exception as e:



def chat_with_ai(user_question, website_text, pdf_text, chat_history):
    # Simple check for common greetings
    greetings = ["hi", "hello", "hey", "hi there", "hello there"]
    if user_question.lower() in greetings:
        return "Hello! How can I assist you today?"

    # If it's a regular question, combine website and PDF content for context
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
            max_tokens=256,
            temperature=0.1,
            stream=False
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating response: {e}"

# ----------------------
# Streamlit UI and App Logic
# ----------------------
st.set_page_config(page_title="AIBYTEC Chatbot", layout="wide")

# Session State Initialization
if "page" not in st.session_state:
    st.session_state['page'] = 'home'
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# ----------------------
# PAGE 1: Home Page with Options
# ----------------------
# if st.session_state['page'] == 'home':
#     # st.title("Welcome to AIByTec Bot")
#     st.subheader("Welcome to AIByTec Bot")
#     st.write("Please choose an option:")




if st.session_state['page'] == 'home':
    st.subheader("Welcome to AIByTec Bot")
    st.write(
        """
        AIByTec Bot is your go-to assistant for enhancing your learning and business strategies. 
        Explore our features by selecting an option below. **Note:** Buttons may require a double-click to function due to certain browser settings.
        """
    )
    st.write("Please choose an option:")
 # Create buttons for the two options
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Fill the Form"):
            st.session_state['page'] = 'form'

    with col2:
        if st.button("Chat with AIByTec Bot"):
            st.session_state['page'] = 'chat'
    


# ----------------------
# PAGE 2: User Info Form
# ----------------------
elif st.session_state['page'] == 'form':
    st.header("Complete Your Profile")

    with st.form(key="user_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        contact_no = st.text_input("Contact No.")    
        specific_needs_and_challenges = st.text_input("Task to be performed")
        training = st.text_input("Preferred course")
        mode_of_training = st.text_input("Online/Onsite")
        prefered_time_contact_mode = st.text_input("Preferred time/mode of contact")

        # Submit Button for the form
        submitted = st.form_submit_button("Submit Profile")
        
        if submitted:
            if name and email and contact_no and specific_needs_and_challenges and training and mode_of_training and prefered_time_contact_mode:
                send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode)
                st.session_state['page'] = 'chat'
                st.success("Your profile has been submitted!")
            else:
                st.warning("Please fill out all fields.")

# ----------------------
# PAGE 3: Chatbot Interface
# ----------------------
elif st.session_state['page'] == 'chat':
    st.header("Chat with AIByTec Bot")

    # Initialize chat history with a greeting from the bot
    if not st.session_state['chat_history']:
        st.session_state['chat_history'].append({
            "user": "", 
            "bot": "Hello! I'm your AI chatbot. How can I assist you today?"
        })

    # Display chat history
    for entry in st.session_state['chat_history']:
        if entry['user']:  # Show user messages
            st.markdown(
                f"""
                <div style="background-color: #439DF6; padding: 10px; color: #fff; border-radius: 10px; margin-bottom: 10px; width: fit-content; max-width: 80%; overflow: hidden;">
                    {entry['user']}
                </div>
                """, 
                unsafe_allow_html=True
            )
        if entry['bot']:  # Show bot messages
            st.markdown(
                f"""
                <div style="background-color: #4a4a4a; padding: 10px; color: #fff; border-radius: 10px; margin-bottom: 10px; margin-left: auto; width: fit-content; max-width: 80%; overflow: hidden;">
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
        st.session_state['chat_history'].append({"user": user_input, "bot": bot_response})'''





# ********************************************************************************************************************************************

'''
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
        mode_of_training = st.text_input("Online/Onsite")  # Updated field
        prefered_time_contact_mode = st.text_input("Preferred time/mode of contact")  # Updated field

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
                </div>
                <div style='display: flex; justify-content: right; margin-bottom: 10px;'>
                <div style='display: flex; align-items: center; max-width: 70%; 
                            background-color:#439DF6; color:rgb(255, 255, 255); 
                            padding: 10px; border-radius: 12px;'>
                    <span>{entry['user']}</span>
                </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        if entry['bot']:  # Show bot messages
            st.markdown(
                f"""
                </div>
                <div style='display: flex; justify-content: left; margin-bottom: 10px;'>
                <div style='display: flex; align-items: center; max-width: 70%; 
                            background-color: #4a4a4a;; color:rgb(255, 255, 255); 
                            padding: 10px; border-radius: 12px;'>
                    <span>{entry['bot']}</span>
                </div>
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
        st.rerun()'''




'''import streamlit as st
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
        mode_of_training = st.text_input("Online/Onsite")  # Updated field
        prefered_time_contact_mode = st.text_input("Preferred time/mode of contact")  # Updated field

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
# PAGE 2: Chatbot Interface with Icons
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
        if entry['user']:  # Show user messages with icon
            st.markdown(
                f"""
                <div style='display: flex; justify-content: right; margin-bottom: 10px;'>
                    <div style='display: flex; align-items: center; max-width: 70%; 
                                background-color:#439DF6; color:rgb(255, 255, 255); 
                                padding: 10px; border-radius: 12px;'>
                        <img src='https://upload.wikimedia.org/wikipedia/commons/6/6c/Octicons-mark-github.svg' 
                             alt='User Icon' style='width: 25px; height: 25px; margin-right: 10px;' />
                        <span>{entry['user']}</span>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        if entry['bot']:  # Show bot messages with icon
            st.markdown(
                f"""
                <div style='display: flex; justify-content: left; margin-bottom: 10px;'>
                    <div style='display: flex; align-items: center; max-width: 70%; 
                                background-color: #4a4a4a;; color:rgb(255, 255, 255); 
                                padding: 10px; border-radius: 12px;'>
                        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Font_Awesome_5_regular_comment-alt.svg/1024px-Font_Awesome_5_regular_comment-alt.svg.png'
                             alt='Bot Icon' style='width: 25px; height: 25px; margin-right: 10px;' />
                        <span>{entry['bot']}</span>
                    </div>
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
        st.rerun()'''


''' 
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
        mode_of_training = st.text_input("Online/Onsite")  # Updated field
        prefered_time_contact_mode = st.text_input("Preferred time/mode of contact")  # Updated field

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
                </div>
                <div style='display: flex; justify-content: right; margin-bottom: 10px;'>
                <div style='display: flex; align-items: center; max-width: 70%; 
                            background-color:#439DF6; color:rgb(255, 255, 255); 
                            padding: 10px; border-radius: 12px;'>
                    <span>👤 {entry['user']}</span>
                </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        if entry['bot']:  # Show bot messages
            st.markdown(
                f"""
                </div>
                <div style='display: flex; justify-content: left; margin-bottom: 10px;'>
                <div style='display: flex; align-items: center; max-width: 70%; 
                            background-color: #4a4a4a;; color:rgb(255, 255, 255); 
                            padding: 10px; border-radius: 12px;'>
                    <span>🤖 {entry['bot']}</span>
                </div>
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
        '''




