import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from data_feed import check_vdb_exists


# Setup for OpenAI
openai_api_key = st.secrets["openai"]
chat = ChatOpenAI(model_name='gpt-4o-mini', temperature=0, openai_api_key=openai_api_key)

# Global variable for the FAISS vectorstore

# Path for the memory file
memory_file_path = "memory.json"
save_directory = "Store"
db = check_vdb_exists(save_directory)

def load_memory():
    if os.path.exists(memory_file_path):
        with open(memory_file_path, "r") as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(memory_file_path, "w") as f:
        json.dump(memory, f)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=10)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

def generate_response(message):
    # Load memory and update it
    memory = load_memory()
    
    best_practice = retrieve_info(message)
    data_csv = "\n".join(best_practice)  # Convert list to string

    # Define a prompt template with a placeholder for data_csv
    prompt_template = PromptTemplate(
        input_variables=["data_csv", "user_message", "memory"],
        template="""
        You are an interviewer assistant. Based on the data I provide and your memory, you will respond accordingly. : {user_message}
        
        data:
        {data_csv}
        
        Memory:
        {memory}
        
        Please provide the most relevant response.
        """
    )
    
    # Initialize LLMChain with the prompt template
    chain = LLMChain(llm=chat, prompt=prompt_template)
    
    # Run the chain with the input values
    response = chain.run(data_csv=data_csv, user_message=message, memory=memory.get('context', ''))
    
    # Update memory with the new interaction
    memory['context'] = response
    save_memory(memory)
    
    return response

# Update CSV from Google Sheets and vectorize it

# Input for the user's question to the LLM
user_input = st.text_input("Ask the LLM a question:", key="user_input")

if user_input:
    # Invoke the LLM with the prompt
    response = generate_response(user_input)
    
    # Display the response from the LLM
    st.write("LLM Response:", response)
