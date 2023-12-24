import streamlit as st
import rag_chatbot_lib as glib

st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")

if 'memory' not in st.session_state:
    st.session_state.memory = glib.get_memory()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_index' not in st.session_state:
    st.session_state.vector_index = glib.get_index()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

input_text = st.chat_input("How can I help you today? ")

if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)
    st.session_state.chat_history.append(
        {"role": "user", "text": input_text})

    chat_response = glib.get_rag_chat_response(
        input_text=input_text, memory=st.session_state.memory, index=st.session_state.vector_index)

    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append(
        {"role": "assistant", "text": chat_response})
