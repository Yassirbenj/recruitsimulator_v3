import streamlit as st
from langchain.llms import OpenAI
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI

# Document data
document2 = {
    "Date": [
        "datetime.datetime(2024, 8, 29, 13, 29, 9, 767619)"
    ],
    "Discussion": [
        """Recruiter: Certainly! Let's start with a foundational question:
        Can you describe a machine learning project you've worked on, detailing the problem you addressed, the approach you took, the data you used, and the outcome of the project?. 
        Candidate: I work in the regulation area for projects, for enemy regulation, and I work in the reinforcement building, like Q-Star Regulation, and I think that's all.. 
        Recruiter: Thank you for sharing that. It sounds like you have experience with reinforcement learning, specifically in a regulatory context. To delve a bit deeper:

        1. Could you explain how you applied Q-learning (or a similar reinforcement learning algorithm) in your regulatory projects? What specific challenges did you face and how did you overcome them?

        2. Additionally, how do you ensure that your reinforcement learning models adhere to regulatory standards and what measures do you implement to verify and validate the models' decisions?"""
    ],
    "Evaluation": [
        """Feedback:

        1. Skills: The candidate mentioned working with reinforcement learning, specifically Q-learning, which is a technical skill relevant to the job's responsibilities. However, the candidate did not provide a detailed explanation of how they applied these skills in their projects. Grade: 60%

        2. Experience: The candidate has experience in the regulation area for projects and reinforcement learning, which aligns with the job role. However, the candidate did not provide specific examples of their previous responsibilities or accomplishments. Grade: 60%

        3. Education Background: The candidate did not mention their educational background, making it difficult to assess how it fits with the required role. Grade: 0%

        4. Relevance of Response: The candidate's response was somewhat relevant, as they mentioned their experience with reinforcement learning. However, they did not fully answer the question, as they did not detail a specific project or the outcome. Grade: 50%

        5. Confidence: The candidate's confidence was not clearly conveyed in the text. However, the candidate did not hesitate to share their experience, which may indicate some level of confidence. Grade: 70%

        6. Language: The candidate used professional language, but the response was not well-structured and lacked detail, making it difficult to fully understand their experience and skills. Grade: 60%

        Global Grade: (60+60+0+50+70+60)/6 = 50%

        The candidate should improve by providing more detailed responses, including specific examples from their experience, and mentioning their educational background. They should also work on structuring their responses more clearly to effectively communicate their skills and experience."""
    ],
    "option": [
        "voice"
    ],
    "full_name": [
        "Adil Boznika"
    ],
    "job_title": [
        "machine learning"
    ],
    "job_details": [
        ""
    ],
    "seniority": [
        "junior"
    ],
    "language": [
        "English"
    ],
    "job_offer": [
        "the title of the job offer is machine learning. the details of the job offer is "
    ]
}

openai_api_key = st.secrets["openai"]
chat = ChatOpenAI(model_name='gpt-4-turbo', temperature=0.1, openai_api_key=openai_api_key)

# Combine document2 parts into a single string
full_document = "\n".join(
    document2["Discussion"] + document2["Evaluation"] + document2["option"] +
    document2["full_name"] + document2["job_title"] + document2["job_details"] +
    document2["seniority"] + document2["language"] + document2["job_offer"]
)

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
all_splits = text_splitter.split_text(full_document)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
save_directory = "Store"

# Load the existing vector store
new_vectorstore = FAISS.load_local(save_directory, OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)

# Add new texts to the vector store
new_vectorstore.add_texts(texts=all_splits, embeddings=embeddings, metadatas=[{}]*len(all_splits))

# Save the updated vector store
new_vectorstore.save_local(save_directory)

# Perform similarity search
question = "List for me all the candidates"
docs = new_vectorstore.similarity_search(question, k=5)

# Concatenate the text from the retrieved documents
context = "\n".join([doc.page_content for doc in docs])

# Use the LLM to generate an answer based on the context
response = chat.invoke(f"Based on the following context, answer the question: {question}\n\nContext:\n{context}")

# Output the answer
st.write("Answer:", response.content)
