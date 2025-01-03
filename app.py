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

# # Load Environment Variables
# load_dotenv()

# SENDER_EMAIL = os.getenv("SENDER_EMAIL")
# SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
# RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
# openai.api_key = os.getenv("OPENAI_API_KEY")
# PDF_PATH = os.getenv("PDF_PATH")
# WEBSITE_URL = os.getenv("WEBSITE_URL")

# # Functions

# # Function to send email
# def send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode):
#     subject = "New User Profile Submission"
#     body = f"""
#     New Student Profile Submitted:
#     Name: {name}
#     Email: {email}
#     Contact No.: {contact_no}
#     Specific Needs & Challenges: {specific_needs_and_challenges}
#     Preferred Course: {training}
#     Mode of Training: {mode_of_training}
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

# # # Function to generate OpenAI response
# # def chat_with_ai(user_question, website_text, pdf_text, chat_history):
# #     combined_context = f"Website Content:\n{website_text}\n\nPDF Content:\n{pdf_text}"
# #     messages = [{"role": "system", "content": "You are a helpful assistant. Use the provided content."}]
# #     for entry in chat_history:
# #         messages.append({"role": "user", "content": entry['user']})
# #         messages.append({"role": "assistant", "content": entry['bot']})
# #     messages.append({"role": "user", "content": f"{combined_context}\n\nQuestion: {user_question}"})

# #     try:
# #         response = openai.ChatCompletion.create(
# #             model="gpt-3.5-turbo",
# #             messages=messages,
# #             max_tokens=256,
# #             temperature=0.5,
# #             stream=False
# #         )
# #         return response['choices'][0]['message']['content']
# #     except Exception as e:



# def chat_with_ai(user_question, website_text, pdf_text, chat_history):
#     # Simple check for common greetings
#     greetings = ["hi", "hello", "hey", "hi there", "hello there"]
#     if user_question.lower() in greetings:
#         return "Hello! How can I assist you today?"

#     # If it's a regular question, combine website and PDF content for context
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
#             temperature=0.1,
#             stream=False
#         )
#         return response['choices'][0]['message']['content']
#     except Exception as e:
#         return f"Error generating response: {e}"

# # ----------------------
# # Streamlit UI and App Logic
# # ----------------------
# st.set_page_config(page_title="AIBYTEC Chatbot", layout="wide")

# # Session State Initialization
# if "page" not in st.session_state:
#     st.session_state['page'] = 'home'
# if "chat_history" not in st.session_state:
#     st.session_state['chat_history'] = []

# # ----------------------
# # PAGE 1: Home Page with Options
# # ----------------------
# # if st.session_state['page'] == 'home':
# #     # st.title("Welcome to AIByTec Bot")
# #     st.subheader("Welcome to AIByTec Bot")
# #     st.write("Please choose an option:")




# if st.session_state['page'] == 'home':
#     st.subheader("Welcome to AIByTec Bot")
#     st.write(
#         """
#         AIByTec Bot is your go-to assistant for enhancing your learning and business strategies. 
#         Explore our features by selecting an option below. **Note:** Buttons may require a double-click to function due to certain browser settings.
#         """
#     )
#     st.write("Please choose an option:")
#  # Create buttons for the two options
#     col1, col2 = st.columns([1, 1])
#     with col1:
#         if st.button("Fill the Form"):
#             st.session_state['page'] = 'form'

#     with col2:
#         if st.button("Chat with AIByTec Bot"):
#             st.session_state['page'] = 'chat'
    


# # ----------------------
# # PAGE 2: User Info Form
# # ----------------------
# elif st.session_state['page'] == 'form':
#     st.header("Complete Your Profile")

#     with st.form(key="user_form"):
#         name = st.text_input("Name")
#         email = st.text_input("Email")
#         contact_no = st.text_input("Contact No.")    
#         specific_needs_and_challenges = st.text_input("Task to be performed")
#         training = st.text_input("Preferred course")
#         mode_of_training = st.text_input("Online/Onsite")
#         prefered_time_contact_mode = st.text_input("Preferred time/mode of contact")

#         # Submit Button for the form
#         submitted = st.form_submit_button("Submit Profile")
        
#         if submitted:
#             if name and email and contact_no and specific_needs_and_challenges and training and mode_of_training and prefered_time_contact_mode:
#                 send_email(name, email, contact_no, specific_needs_and_challenges, training, mode_of_training, prefered_time_contact_mode)
#                 st.session_state['page'] = 'chat'
#                 st.success("Your profile has been submitted!")
#             else:
#                 st.warning("Please fill out all fields.")

# # ----------------------
# # PAGE 3: Chatbot Interface
# # ----------------------
# elif st.session_state['page'] == 'chat':
#     st.header("Chat with AIByTec Bot")

#     # Initialize chat history with a greeting from the bot
#     if not st.session_state['chat_history']:
#         st.session_state['chat_history'].append({
#             "user": "", 
#             "bot": "Hello! I'm your AI chatbot. How can I assist you today?"
#         })

#     # Display chat history
#     for entry in st.session_state['chat_history']:
#         if entry['user']:  # Show user messages
#             st.markdown(
#                 f"""
#                 <div style="background-color: #439DF6; padding: 10px; color: #fff; border-radius: 10px; margin-bottom: 10px; width: fit-content; max-width: 80%; overflow: hidden;">
#                     {entry['user']}
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
#         if entry['bot']:  # Show bot messages
#             st.markdown(
#                 f"""
#                 <div style="background-color: #4a4a4a; padding: 10px; color: #fff; border-radius: 10px; margin-bottom: 10px; margin-left: auto; width: fit-content; max-width: 80%; overflow: hidden;">
#                     {entry['bot']}
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
    
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





# ********************************************************************************************************************************************


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
                <div style="
                    background-color: #439DF6; 
                    padding: 10px;
                    color: #fff;
                    border-radius: 10px; 
                    # margin-bottom: 10px;
                    # width: fit-content;
                    max-width: 70%;
                    overflow: hidden;
                    # display: flex; 
                    # align-items: center;
                    border-radius: 10px;
                ">
        
                    {entry['user']}
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
                <div style="
                    background-color: #4a4a4a; 
                    padding: 10px; 
                    color: #fff; 
                    border-radius: 10px; 
                    # margin-bottom: 10px;
                    # margin-left: auto;
                    width: fit-content;
                    max-width: 70%;
                    # overflow: hidden;
                    # display: flex;
                    # align-items: end;
                ">

                    {entry['bot']}
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

