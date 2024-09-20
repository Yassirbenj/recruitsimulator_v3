from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from io import StringIO

import streamlit as st
from streamlit_chat import message
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from datetime import datetime
from config_grid import scoring_eval


st.set_page_config(page_title="Customer simulator ")
st.header("Customer simulator")

def main():
    openai_api_key = st.secrets["openai"]

    chat=ChatOpenAI(model_name='gpt-4',temperature=0.5,openai_api_key=openai_api_key)

    if "messages" not in st.session_state:
        st.cache_data.clear()
        personae=config_persona()
        eval_grid=scoring_eval()
        if personae:
            st.session_state.messages=[
                SystemMessage(content=personae)
                ]
            st.session_state.cost=0
            st.session_state.evals=[]
        if eval_grid:
            st.session_state.grid=eval_grid

    if prompt := st.chat_input("Start your call with an introduction"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.sidebar:
                evaluation=evaluate_sentence(prompt)
                st.write(evaluation)
                st.session_state.evals.append({prompt:evaluation})
        with st.spinner ("Thinking..."):
            with get_openai_callback() as cb:
                response=chat(st.session_state.messages)
                st.session_state.cost=round(cb.total_cost,5)
        st.session_state.messages.append(AIMessage(content=response.content))

    messages=st.session_state.get('messages',[])
    discussion=""

    for i,msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content,is_user=True,key=str(i)+'_saleperson')
            discussion+=f"Sale person: {msg.content}. "
        else:
            message(msg.content,is_user=False,key=str(i)+'_customer')
            discussion+=f"Customer: {msg.content}. "

    if len(messages) > 5:
        evaluate_button=st.button("Evaluate")
        if evaluate_button:
            if discussion=="":
                st.write("No discussion to evaluate")
            elif len(messages) <= 5:
                st.write("The discussion is too short to be evaluated")
            else:
                recap_response=recap(discussion)
                evaluation_response=evaluate(discussion,st.session_state.grid)
                st.title("Recommendations")
                evaluations=st.session_state.evals
                st.write(evaluations)
                st.cache_data.clear()
                conn = st.experimental_connection("gsheets", type=GSheetsConnection, ttl=1)
                df=conn.read()
                last_index=df.iloc[-1,0]

                #get current time
                current_datetime = datetime.now()

                data={
                        "Index":[last_index+1],
                        "User":[""],
                        "Date":[current_datetime],
                        #"Personae":[st.session_state.personae],
                        "Discussion":[discussion],
                        "Evaluation":[evaluation_response],
                        "Recap":[recap_response]
                    }

                data_df=pd.DataFrame(data)
                data_df_updated=pd.concat([df,data_df])
                conn.update(worksheet="evals",data=data_df_updated)
                st.write("Evaluation stored with success")


def recap(discussion):
    openai_api_key = st.secrets["openai"]
    llm=OpenAI(openai_api_key=openai_api_key)
    template = """Question: summarize the discussion between customer and sales person based on following discussion {question} """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response=llm_chain.run(discussion)
    st.title("Recap of the discussion")
    st.write(response)
    return response

def evaluate_sentence(sentence):
    openai_api_key = st.secrets["openai"]
    llm=OpenAI(openai_api_key=openai_api_key)
    template = """Question: you are a coach for sales persons. this sentence {question} is from a sales person discussing with a customer. do you have a better formulation that will help to improve the sales process?  explain why"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response=llm_chain.run(sentence)
    return response

def evaluate(discussion,grid):
    openai_api_key = st.secrets["openai"]
    llm=OpenAI(openai_api_key=openai_api_key)
    template = """Question: evaluating a discussion between a sales person and a customer based on following discussion {discussion}. give a feedback to the sales person on the good points and the major point to be improved based on the following evaluation grid {grid} """
    prompt = PromptTemplate(template=template, input_variables=["discussion","grid"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    input_list = {"discussion": discussion,"grid": grid}
    response=llm_chain.run(input_list)
    st.title("Evaluation of the discussion")
    st.write(response)
    return response

def config_persona():
    product=st.text_input("What category of product are you selling (ex: CRM, aluminium windows...) ?")
    product_name=st.text_input("whats the name of your product ?")
    #type_customer=st.selectbox("Are you selling to a company or a direct consumer ?",('B2B'))
    industry=st.text_input("To what industry do you want to sell (ex: hotels, construction ) ?")
    department=st.text_input("What department within the company are you calling (ex: finance, operations...) ?")
    reason=st.selectbox("did the customer contacted you before ?",('no','yes, fulfill a contact form','yes, contacted elsewhere'))
    personnality=st.text_input("what are the main traits of character of your contact person (ex: busy, willing to discuss, impolite...) ?")
    start=st.button("start")
    if start:
        #context
        openai_api_key = st.secrets["openai"]
        llm=OpenAI(openai_api_key=openai_api_key)
        template = """Question: if you are working in department {department} of a company in the industry {industry}, what will be the main points you want to check before buying a product type {product} ?"""
        prompt = PromptTemplate(template=template, input_variables=["department","industry","product"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        input_list = {"department": department,"industry": industry,"product": product}
        context=llm_chain(input_list)
        context_text=context["text"]
        #competitors
        llm2=OpenAI(openai_api_key=openai_api_key)
        template2 = """Question: if you are working in a company in the industry {industry}, what will be the main products of type {product} that compete with product {product_name} ?"""
        prompt2 = PromptTemplate(template=template2, input_variables=["industry","product","product_name"])
        llm_chain2 = LLMChain(prompt=prompt2, llm=llm2)
        input_list2 = {"industry": industry,"product": product,"product_name":product_name}
        competition=llm_chain2(input_list2)
        competition_text=competition["text"]

        persona="You are a customer responding to a call from a sales person. "
        persona+="You are in the industry of "+ industry + "."
        persona+=" your main personality trait are "+ personnality +"."
        persona+="you will try to understand what the sales person have to offer. asking pertinent questions about the product"
        persona += f"You will try to evaluate the sales person proposition based on following main points : {context_text}. you will try to validate one point after the other."
        persona += f"before concluding You will try to challenge the sales persons about their competitors: {competition_text}. you will ask the question after understanding the sales person offer"
        persona += "You respond briefly to the questions. you do not easily disclose your needs and expectations easily. you are a customer not an assistant "
        if reason=="'no":
            persona+="you never contacted this company before. you need to understand who's calling and why before going forward."
        elif reason=='yes, fulfill a contact form':
            persona+="you have fulfilled a contact form in the company's website."
        else:
            persona+="you have been in contact with this company elsewhere."
        return persona

def scoring_eval():
    uploaded_file = st.file_uploader("Choose a evaluation grid file")
    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        decoded_response = file_contents.decode('utf-8')
        #st.write(decoded_response)
        return decoded_response

#config_persona()
main()
#scoring_eval()
