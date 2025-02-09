import os
import random
import streamlit as st

from graph import FirstAgent

# Set environment variables for Langchain and SendGrid
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"]="FirstAgent"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"

DEBUGGING=0

# This function sets up the chat interface and handles user interactions
def start_chat():
    
    # Setup a simple landing page with title and avatars
    st.title('SQL Translator')
    st.text("""Please ask any questions on the database schema. 
               I can also amend results based on follow up questions.
            """)
    avatars={"system":"💻🧠","user":"🧑‍💼","assistant":"🎓", "result":"🎓"}
    
    # Keeping context of conversations, checks if there is anything in messages array
    # If not, it creates an empty list where all messages will be saved
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Ensuring a unique thread-id is maintained for every conversation
    if "thread-id" not in st.session_state:
        st.session_state.thread_id = random.randint(1000, 9999)
    thread_id = st.session_state.thread_id

    # Display previous messages in the chat history by keeping track of the messages array
    # in the session state. 
    for message in st.session_state.messages:
        if message["role"] != "system":
            avatar=avatars[message["role"]]
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"]) 

    # Handle new user input. Note: walrus operator serves two functions, it checks if
    # the user entered any input. If yes, it returns that value and assigns to 'prompt'. Note that escaped_prompt was
    # used for formatting purposes.
    if prompt := st.chat_input("What's up?"):
        escaped_prompt = prompt.replace("$", "\\$")
        st.session_state.messages.append({"role": "user", "content": escaped_prompt})
        with st.chat_message("user", avatar=avatars["user"]):
            st.write(escaped_prompt)
        
        # Initialize salesCompAgent in graph.py 
        app = FirstAgent(st.secrets['OPENAI_API_KEY'])
        thread={"configurable":{"thread_id":thread_id}}
        
        # Stream responses from the instance of salesCompAgent which is called "app"
        for s in app.graph.stream({'initialMessage': prompt, 'sessionState': st.session_state}, thread):
    
            if DEBUGGING:
                print(f"GRAPH RUN: {s}")
                st.write(s)
            for k,v in s.items():
                if DEBUGGING:
                    print(f"Key: {k}, Value: {v}")
            if resp := v.get("responseToUser"):
                with st.chat_message("assistant", avatar=avatars["assistant"]):
                    print (resp)
                    st.text(resp) 
                st.session_state.messages.append({"role": "result", "content": resp})

if __name__ == '__main__':
    start_chat()