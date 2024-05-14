import streamlit as st
import os
import shutil
from myImg import img
import streamlit.components.v1 as components
import time
from ollama_model import document_loader, embedding_model, vector_db
from retriever import response
import random


st.set_page_config(page_title="🦙💬 Climate Change LLM")

components.html(f'<div style="color:#3399FF;font-size:36px; display:flex; gap:1rem"><img style="height:15rem" src="{img}"/><h1 style="font-size:4rem;color:rgb(255, 75, 75) ">Climate Change LLM</h1></div>', width=None, height=227, scrolling=False)

models = {
    "Llama3-8B": "meta/meta-llama-3-8b-instruct",
    "Mistral-7B": "mistralai/mistral-7b-instruct-v0.2",
    "Llama2-7B": "meta/llama-2-7b-chat"
}

REPLICATE_API_TOKEN= "r8_UhPq7KWotNkc8JscYADAfA8pzOfiZxH048way"
# os.environ['REPLICATE_API_TOKEN'] = "r8_UhPq7KWotNkc8JscYADAfA8pzOfiZxH048way"

def select_model():
    return st.sidebar.selectbox('Choose a Model', list(models.keys()))



def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

@st.cache_data
def load_files(files):
    data_folder = 'data'
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    os.makedirs(data_folder)

    file_paths = []
    for file in files:
        file_path = os.path.join(data_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        file_paths.append(file_path)

    return data_folder

def main(initial):
    with st.sidebar:
        st.subheader('Models and parameters')
        selected_model = select_model()
        llm = models[selected_model]

        st.button('Clear Chat History', on_click=clear_chat_history)

        file_types = ["pdf", "txt", "md", "html", "xlsx", "xls", "csv", "doc", "docx"]
        files = st.file_uploader("Upload multiple files to chat with them.", accept_multiple_files=True, type=file_types)
    

    if files and 'count' not in st.session_state:
        if 'count' not in st.session_state:
            st.session_state.count = 0
        data_folder = load_files(files)

        all_splits = document_loader(data_folder)
        a = st.success("Files Loaded Successfully!", icon='✅')
        initial=True
        time.sleep(4)
        a.empty()
        embeddings = embedding_model()

        # Create vector db
        random_dir = "d" + str(random.randint(100, 999))
        st.session_state.persist_directory = f"qdrant_db/{random_dir}"
        vectordb=vector_db(all_splits, embeddings, st.session_state.persist_directory)
        b = st.success("Vector Database Loaded Successfully!", icon='✅')
        time.sleep(4)
        b.empty()
        st.session_state.retreiver = vectordb.as_retriever()
    if 'count' in st.session_state:
        prompt = st.chat_input("Type your message here")
        # persist_directory = "qdrant_db/d3"
        retriever = st.session_state.retreiver
        if prompt:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        output = response(retriever, prompt, st.session_state.persist_directory, llm)
                        placeholder = st.empty()
                        full_response = ''
                        for item in output:
                            full_response += item
                            placeholder.markdown(full_response)
                        placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "user", "content": prompt})
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)
             
    display_chat_messages()


if __name__ == "__main__":
    main(0)
