import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PyPDF2 import PdfReader
import openai
import os

# ----------------------
# Set Page Configuration (Must Be First Streamlit Command)
# ----------------------
st.set_page_config(page_title="Student Profile & AI Chatbot", layout="wide")

# ----------------------
# Load Environment Variables
# ----------------------
# Assuming you have environment variables set for sensitive data
# SENDER_EMAIL = os.getenv("SENDER_EMAIL")
# SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
# RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Hard-code the PDF path
predefined_pdf_path = "./Aibytec fine tuned data.pdf"  # Replace with your actual PDF file path

# ----------------------
# Functions
# ----------------------

# Extract Text from Hard-Coded PDF
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

# Function to generate OpenAI response
def chat_with_ai(user_question, chat_history, pdf_text):
    # Initial greeting message
    initial_greeting = "Hello! How can I assist you today?"
    messages = [{"role": "system", "content": "You are a helpful assistant. please greet the user first and then Use the provided content."}]
    
    # Add greeting message if chat history is empty
    if len(chat_history) == 0:
        messages.append({"role": "assistant", "content": initial_greeting})
    
    for entry in chat_history:
        messages.append({"role": "user", "content": entry['user']})
        messages.append({"role": "assistant", "content": entry['bot']})
    messages.append({"role": "user", "content": f"{pdf_text}\n\nQuestion: {user_question}"})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=256,
            temperature=0.7,
            stream=False
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating response: {e}"

# ----------------------
# Streamlit UI and App Logic
# ----------------------

# Session State Initialization
if "page" not in st.session_state:
    st.session_state['page'] = 'form'
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# ----------------------
# PAGE 1: User Info Form
# ----------------------
if st.session_state['page'] == 'form':
    with st.form(key="user_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        contact_no = st.text_input("Contact No.")
        area_of_interest = st.text_input("Area of Interest")
        
        # Create two columns for buttons
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Proceed to Chat ")
        with col2:
            continue_chat = st.form_submit_button(" Skip and Join Chat")
        
        if submitted:
            if name and email and contact_no and area_of_interest:
                # You could also add a function to send an email if needed
                st.session_state['page'] = 'chat'
                st.rerun()
            else:
                st.warning("Please fill out all fields.")
        
        # If user clicks "Continue Chat with AIByTec"
        if continue_chat:
            st.session_state['page'] = 'chat'
            st.rerun()

# ----------------------
# PAGE 2: Chatbot Interface
# ----------------------
elif st.session_state['page'] == 'chat':
    # Display chat history without headings
    for entry in st.session_state['chat_history']:
        # User Message
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

        # Assistant Message
        st.markdown(
            f"""
            <div style="
                background-color:  #4a4a4a; 
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

    # Use the predefined PDF content
    if os.path.exists(predefined_pdf_path):
        pdf_text = extract_pdf_text(predefined_pdf_path)
    else:
        pdf_text = "PDF content not loaded."

    # Fixed input bar at bottom
    user_input = st.chat_input("Type your question here...", key="user_input_fixed")

    if user_input:
        # Display bot's response
        with st.spinner("Generating response..."):
            bot_response = chat_with_ai(user_input, st.session_state['chat_history'], pdf_text)
        
        # Append user query and bot response to chat history
        st.session_state['chat_history'].append({"user": user_input, "bot": bot_response})
        
        # Re-run to display updated chat history
        st.rerun()
