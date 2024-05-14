import streamlit as st
from ollama_model import embedding_model
import replicate
from langchain_community.vectorstores import Qdrant

api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
def generate_llama2_response(context,question,model_name):
    formatted_prompt =  f"""
                Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
                provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
                Context:\n {context}\n
                Question: \n{question}\n

                Answer:
            """


    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            formatted_prompt += "User: " + dict_message["content"] + "\n\n"
        else:
            formatted_prompt += "Assistant: " + dict_message["content"] + "\n\n"
    output = api.run(model_name, 
                           input={"prompt": f"{formatted_prompt} Assistant: ","repetition_penalty":1,"max_length":1024})

    return output


# RAG Setup
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(retriever, question, persist_directory, model_name):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return generate_llama2_response(question, formatted_context, model_name)

def response(retriever, question, persist_directory, model_name):
    result = rag_chain(retriever, question, persist_directory, model_name)
    return result
