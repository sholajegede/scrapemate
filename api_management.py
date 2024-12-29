import streamlit as st
import os

def get_api_key(api_key_name):
    if api_key_name == 'GROQ_API_KEY':
        return st.session_state['groq_api_key'] or os.getenv(api_key_name)
    else:
        return os.getenv(api_key_name)
        