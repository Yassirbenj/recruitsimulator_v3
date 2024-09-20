from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]

def update_csv_from_gsheets(worksheet_name, csv_file_name):
    # Establish the connection to Google Sheets
    conn = st.connection("gsheets", type=GSheetsConnection, ttl=1)
    
    # Read existing data from the Google Sheet
    df = conn.read(worksheet=worksheet_name)
    
    # Check if the file already exists
    if os.path.exists(csv_file_name):
        # Delete the old file
        os.remove(csv_file_name)
        print(f"Old file '{csv_file_name}' deleted.")

    # Save the DataFrame to a new CSV file
    df.to_csv(csv_file_name, index=False)
    print(f"New file '{csv_file_name}' created.")

    return df

# Update the CSV file and load initial data
#update_csv_from_gsheets("entretiens", "google_sheet_data.csv")

# Global variable for the FAISS database

save_directory = "Store"

    


def check_vdb_exists(file_path):
    if os.path.exists(file_path):
        print(f"The file '{file_path}' already exists.")
        db = FAISS.load_local(file_path, OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
        return db
    else:
        print(f"The file '{file_path}' does not exist. Continuing with the code...")
        update_csv_from_gsheets("entretiens", "google_sheet_data.csv")
        loader = CSVLoader(file_path="google_sheet_data.csv")
        documents = loader.load()
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        #all_splits = []
        #for doc in documents:
            #splits = text_splitter.split_text(doc.page_content)
            #all_splits.extend(splits)
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a FAISS vectorstore from the loaded documents
        db = FAISS.from_documents(documents=documents, embedding=embeddings)
        db.save_local(file_path)
        return db





# Function to retrieve relevant information based on a query
def retrieve_info(query):
    db = check_vdb_exists(save_directory)
    similar_response = db.similarity_search(query, k=2)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# Initialize ChatOpenAI model
llm = ChatOpenAI(model_name='gpt-4-turbo', temperature=0, openai_api_key=openai_api_key)

# Define the prompt template
template = """You are a recruiter conducting an interview for the job offer. I will share our candidate information that you have interviewed, and you will help manage their information like their name, grade, and position they applied for. Below is the message you received: {message}. Here is the list of the data: {data_csv}. Please work as an assistant to help the company manage the data."""
prompt = PromptTemplate(
    input_variables=["message", "data_csv"],
    template=template
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Function to generate a response from the LLM
def generate_response(message):
    best_practice = retrieve_info(message)
    data_csv = "\n".join(best_practice)  # Convert list to string
    response = chain.run(message=message, data_csv=data_csv)
    return response

# Example usage
