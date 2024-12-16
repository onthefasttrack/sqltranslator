import streamlit as st
import oracledb
from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

class AgentState(TypedDict):
    agent: str
    initialMessage: str
    responseToUser: str
    lnode: str
    category: str
    sessionState: Dict

class Category(BaseModel):
    category: str

class SQL(BaseModel):
    sql_present: bool
    sql_query: str

def create_llm_message(system_prompt):
    # Initialize empty list to store messages
    resp = []
    
    # Add system prompt as the first message. This will provide the overall instructions to LLM.
    resp.append(SystemMessage(content=system_prompt))
    
    # Get chat history from Streamlit's session state
    msgs = st.session_state.messages
    
    # Iterate through chat history, and based on the role (user or assistant) tag it as HumanMessage or AIMessage
    for m in msgs:
        if m["role"] == "user":
            # Add user messages as HumanMessage
            resp.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            # Add assistant messages as AIMessage
            resp.append(AIMessage(content=m["content"]))
    
    # Return the formatted message list
    return resp

def ObtainSchema():
        connection=oracledb.connect(
            config_dir="/workspaces/sqltranslator/.streamlit",
            user="admin",
            password=st.secrets['ADMIN_PASS'],
            dsn="db1_low",
            wallet_location="/workspaces/sqltranslator/.streamlit",
            wallet_password=st.secrets['WALLET_PASS'])
    
        cursor = connection.cursor()
        resp = "The database schema is :"
        for row in cursor.execute("select table_name, column_name, data_type from user_tab_columns"):
            resp += f"{row} \n"
        print (resp)
        connection.close()
        return resp

class FirstAgent:
    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

        workflow = StateGraph(AgentState)

        workflow.add_node("classifier",self.classifier)
        workflow.add_node("movies",self.MovieAgent)
        workflow.add_node("others",self.otherAgent)
        workflow.add_node("catchall",self.catchallAgent)

        workflow.add_edge(START, "classifier")
        workflow.add_conditional_edges("classifier", self.main_router)
        workflow.add_edge("movies", END)
        workflow.add_edge("others", END)
        workflow.add_edge("catchall", END)


        self.graph = workflow.compile()



    def classifier(self, state: AgentState):
        CLASSIFIER_PROMPT=f"""
        You are an expert with deep knowledge of Movies. 
        Your job is to comprehend the message from the user even if it lacks specific keywords, 
        always maintain a friendly, professional, and helpful tone. 
        If a user greets you, greet them back by mirroring user's tone and verbosity, and offer assitance. 

        Based on user query, accurately classify customer requests into one of the following categories based on context 
        and content, even if specific keywords are not used.
        1. Movie Question: set category as "MOVIES"
        2. Other: set category as "OTHERS"
        """
        llm_messages = create_llm_message(CLASSIFIER_PROMPT)
        llm_response = self.model.with_structured_output(Category).invoke(llm_messages)

        category = llm_response.category
        print(f"{category=}, {llm_response=}")
        return {"category": category}

    def main_router(self, state: AgentState):
        print(f"{state=}")
        category = state.get("category")
        print(f"{category=}")
        if category == "MOVIES":
            return "movies"
        elif category == "OTHERS":
            return "others"
        else:
            return "catchall"
    

    def MovieAgent(self, state: AgentState):
        schema = ObtainSchema()
        SQL_PROMPT=f"""
        You are an expert with deep knowledge of Oracle SQL. 
        Your job is to create a sql query to answer the question the user posts. 
        The Database schema is {schema}
        If there is no sql query for the user request, then mark sql_present as false.
        """
        llm_messages = create_llm_message(SQL_PROMPT)
        llm_response = self.model.with_structured_output(SQL).invoke(llm_messages)
        sql_present = llm_response.sql_present
        sql_query   = llm_response.sql_query
        return {"responseToUser": f"This is a movie agent with {sql_present=}, {sql_query=}"}

    def otherAgent(self, state: AgentState):
        return {"responseToUser": "This is an other agent"}

    def catchallAgent(self, state: AgentState):
        return {"responseToUser": "This is a catchall agent"}