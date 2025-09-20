# frontend/app.py
import streamlit as st
import requests
import uuid
import sys
import os

# --- Add Project Root to Python Path ---
# This is the crucial change to fix the import error
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now this import will work correctly
from config import API_URL

# --- Page Configuration ---
st.set_page_config(
    page_title="FXEC Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("Language Agnostic Chatbot 🤖")
st.caption("Your friendly assistant for queries about the Francis Xavier Engineering College.")

# --- Session State Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Logic ---
if prompt := st.chat_input("Ask me anything about the college..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    data = {"query": prompt, "session_id": st.session_state.session_id}

    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, data=data)
            response.raise_for_status()
            
            api_response = response.json()
            answer = api_response.get("answer", "Sorry, I couldn't get an answer.")
            sources = api_response.get("sources", [])
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources):
                            st.write(f"**Source {i+1}:** {source['metadata'].get('source', 'N/A')}")
                            st.write(source['content'])
                            st.divider()

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the API. Please make sure the backend is running. Error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")