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

class Result(BaseModel):
   result_text: str

def create_llm_message(system_prompt):
    # Initialize empty list to store messages
    resp = []
    
    # Add system prompt as the first message. This will provide the overall instructions to LLM.
    resp.append(SystemMessage(content=system_prompt))
    
    # Get chat history from Streamlit's session state
    msgs = st.session_state.messages[-4:]
    
    # Iterate through chat history, and based on the role (user or assistant) tag it as HumanMessage or AIMessage
    for m in msgs:
      if m["role"] == "user":
        # Add user messages as HumanMessage
        resp.append(HumanMessage(content=m["content"]))
      elif m["role"] == "assistant":
        # Add assistant messages as AIMessage
        resp.append(AIMessage(content=m["content"]))
      elif m["role"] == "result":
        # Add assistant messages as AIMessage
        resp.append(AIMessage(content=m["content"]))
    
    #resp.append(HumanMessage(content=msgs[0]["content"]))
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
        #obtain top tables for the user query
        msgs = st.session_state.messages
    
        # Look at the most recent human message and find relevant table names
        m = msgs[-1]["content"]
        print (f"{m=}")
        query = """SELECT
    '# CREATE TABLE "' || atc.owner || '"."' || atc.table_name || '" ' ||
    '(' ||
    LISTAGG('"' || atc.column_name || '" ' || atc.data_type ||
      CASE
        WHEN atc.data_type IN ('CHAR', 'VARCHAR2', 'NCHAR', 'NVARCHAR2') THEN
          '(' || atc.data_length || ')'
        WHEN atc.data_type = 'NUMBER' AND
            (atc.data_precision IS NOT NULL OR atc.data_scale IS NOT NULL) THEN
          '(' || COALESCE(atc.data_precision, 38) || ',' ||
          COALESCE(atc.data_scale, 0) || ')'
        ELSE NULL
      END || ' ''' || REPLACE(acc.comments, '''', '''''') || '''', ',') || ')' AS table_definition
FROM sys.all_tab_columns atc LEFT JOIN sys.all_col_comments acc
ON atc.owner = acc.owner AND 
   atc.table_name = acc.table_name AND
   atc.column_name = acc.column_name
WHERE atc.owner = 'ADMIN' AND
      atc.table_name in (select table_name from tabnames order by
                     vector_distance(vec, TO_VECTOR(VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING :search_text as DATA)), COSINE)
                      fetch approx first 5 rows only)
GROUP BY atc.owner, atc.table_name"""
        #query = """select table_name from tabnames order by
        #             vector_distance(vec, TO_VECTOR(VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING :search_text as DATA)), COSINE)
        #              fetch approx first 3 rows only"""
        #query = "select table_name, column_name, data_type from user_tab_columns"
        
        resp = "The tables are :"
        for row in cursor.execute(query, search_text=m):
            resp += f"{row} \n"
        print (resp)
        connection.close()
        return resp

def ObtainFeedback():
        connection=oracledb.connect(
            config_dir="/workspaces/sqltranslator/.streamlit",
            user="admin",
            password=st.secrets['ADMIN_PASS'],
            dsn="db1_low",
            wallet_location="/workspaces/sqltranslator/.streamlit",
            wallet_password=st.secrets['WALLET_PASS'])
    
        cursor = connection.cursor()
        #obtain top tables for the user query
        msgs = st.session_state.messages
    
        # Look at the most recent human message and find relevant table names
        m = msgs[-1]["content"]
        print (f"{m=}")
        query = """select user_query || ':' || comments from user_feedback order by
                     vector_distance(vec, TO_VECTOR(VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING :search_text as DATA)), COSINE)
                      fetch approx first 3 rows only"""
        
        resp = "The feedback data is :"
        for row in cursor.execute(query, search_text=m):
            resp += f"{row} \n"
        print (resp)
        connection.close()
        return resp

def queryDBResults(sql_query: str):
      connection=oracledb.connect(
            config_dir="/workspaces/sqltranslator/.streamlit",
            user="admin",
            password=st.secrets['ADMIN_PASS'],
            dsn="db1_low",
            wallet_location="/workspaces/sqltranslator/.streamlit",
            wallet_password=st.secrets['WALLET_PASS'])
      cursor = connection.cursor()
      query_to_issue = sql_query.rstrip(';')
      query = f"""SELECT JSON_ARRAYAGG(JSON_OBJECT(* ABSENT ON NULL RETURNING CLOB) ABSENT ON NULL RETURNING CLOB PRETTY) FROM (
    {query_to_issue})"""
      print(f"{query=}")
      try:
        cursor.execute(query)
        while True:
          row = cursor.fetchone()
          if row is None:
            break
          result = str(row[0])
          print(row[0])
          results_present = True
      except oracledb.Error as e:
        error_obj, = e.args
        print("Error Code:", error_obj.code)
        print("Error Full Code:", error_obj.full_code)
        print("Error Message:", error_obj.message)
      finally:
        connection.close()

      return results_present, result

class FirstAgent:
    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

        workflow = StateGraph(AgentState)

        workflow.add_node("classifier",self.classifier)
        workflow.add_node("manufacturing",self.ManufacturingAgent)
        workflow.add_node("feedback",self.FeedbackAgent)
        workflow.add_node("others",self.otherAgent)
        workflow.add_node("catchall",self.catchallAgent)

        workflow.add_edge(START, "classifier")
        workflow.add_conditional_edges("classifier", self.main_router)
        workflow.add_edge("manufacturing", END)
        workflow.add_edge("feedback", END)
        workflow.add_edge("others", END)
        workflow.add_edge("catchall", END)
        self.graph = workflow.compile()


    def classifier(self, state: AgentState):
        CLASSIFIER_PROMPT=f"""
        You are an expert with deep knowledge of product, supply chain management, sales of products
        Your job is to comprehend the message from the user even if it lacks specific keywords, 
        always maintain a friendly, professional, and helpful tone. 
        If a user greets you, greet them back by mirroring user's tone and verbosity, and offer assitance. 

        Based on user query, accurately classify customer requests into one of the following categories based on context 
        and content, even if specific keywords are not used.
        1. Feedback based on the previous messages or a request to amend or modify the results : set category as "feedback"
        2. Manufacturing or inventory or sales or warehouses or suppliers or employees or consumer data: set category as "manufacturing"
        3. Other: set category as "OTHERS"
        """
        llm_messages = create_llm_message(CLASSIFIER_PROMPT)
        print (f"{llm_messages=}")
        llm_response = self.model.with_structured_output(Category).invoke(llm_messages)

        category = llm_response.category
        print(f"{category=}, {llm_response=}")
        return {"category": category}

    def main_router(self, state: AgentState):
        print(f"{state=}")
        category = state.get("category")
        print(f"{category=}")
        if category == "manufacturing":
            return "manufacturing"
        elif category == "feedback":
            return "feedback"
        elif category == "OTHERS":
            return "others"
        else:
            return "catchall"

    def ManufacturingAgent(self, state: AgentState):
        schema = ObtainSchema()
        feedback = ObtainFeedback()
        
        SQL_PROMPT = f"""Given an input Question, create a syntactically correct Oracle SQL query to run.
        - Pay attention to using only the column names that you can see in the schema description.
        - Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        - Please double check that the SQL query you generate is valid for Oracle Database.
        - DO NOT write anything else except the Oracle SQL.
        - If there is no sql query for the user request, then mark sql_present as false.
        - Only use the tables listed below {schema}
        - Incorporate the feedback given by user before {feedback}
        """
        llm_messages = create_llm_message(SQL_PROMPT)
        llm_response = self.model.with_structured_output(SQL).invoke(llm_messages)
        sql_present = llm_response.sql_present
        sql_query   = llm_response.sql_query
        results_present = False
        if sql_present:
          results_present, result = queryDBResults(sql_query)
          
          if results_present:
            RESULTS_PROMPT = f"""Given the input Question, if the results in json format are {result}
                                 convert the above to be more readable answer to the user. Just only give the answer
                                 Please dont eliminate any data from the input json. Show all data given.
                              """
            llm_messages = create_llm_message(RESULTS_PROMPT)
            llm_response = self.model.with_structured_output(Result).invoke(llm_messages)
            result_text = llm_response.result_text
            return {"responseToUser": f"{result_text} \n The sql query used is - \n {sql_query}"}
          else:
            FIXQUERY_PROMPT = f"""Please fix the query {sql_query} given the question as it resulted 
                                  in an error {error_obj.message} when querying the oracle database
                               """ 
            llm_messages = create_llm_message(SQL_PROMPT)
            llm_response = self.model.with_structured_output(SQL).invoke(llm_messages)
            sql_present = llm_response.sql_present
            sql_query   = llm_response.sql_query
            results_present, result = queryDBResults(sql_query)
            if results_present:
              RESULTS_PROMPT = f"""Given the input Question, if the results in json format are {result}
                                   convert the above to be more readable answer to the user. Just only give the answer
                                   Please dont eliminate any data from the input json. Show all data given.
                                """
              llm_messages = create_llm_message(RESULTS_PROMPT)
              llm_response = self.model.with_structured_output(Result).invoke(llm_messages)
              result_text = llm_response.result_text
              return {"responseToUser": f"{result_text} \n The fixed sql query used is - \n {sql_query}"}
            else:
              return {"responseToUser": f"Unable to generate results for sql query - {sql_query}"}
        else:
          return {"responseToUser": f"Could not generate the sql for this question. Please try a different question"}

    def FeedbackAgent(self, state: AgentState):
        #save the feedback in user_feedback table
        connection=oracledb.connect(
            config_dir="/workspaces/sqltranslator/.streamlit",
            user="admin",
            password=st.secrets['ADMIN_PASS'],
            dsn="db1_low",
            wallet_location="/workspaces/sqltranslator/.streamlit",
            wallet_password=st.secrets['WALLET_PASS'])
    
        cursor = connection.cursor()
        #obtain top tables for the user query
        msgs = st.session_state.messages
    
        # Look at the most recent human message and find relevant table names
        mcomment = msgs[-1]["content"]
        # Get chat history from Streamlit's session state
        for m in msgs[::-2]:
          if m["role"] == "user":
            # Add user messages as HumanMessage
            userq = m["content"]

        if userq is not None:
          print(f"{userq=} {mcomment=}")
          print(userq)
          cursor.execute("insert into user_feedback values (:1, :2, VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING :3 as DATA))", [mcomment, userq, userq])
          connection.commit()
          connection.close()
          #remove the feedback message
          st.session_state.messages.pop()
          return self.ManufacturingAgent(state)
        else:
           return {"responseToUser": "Thanks for the feedback"}
    
    def otherAgent(self, state: AgentState):
        return {"responseToUser": "This is an other agent"}

    def catchallAgent(self, state: AgentState):
        return {"responseToUser": "This is a catchall agent"}