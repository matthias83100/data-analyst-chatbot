import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Streamlit App Title
st.title("LangChain Chatbot")

# Initialize LangChain Groq Client
groq_api_key = st.secrets["GROQ_API_KEY"]  
model = "llama3-8b-8192"
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

# System prompt
system_prompt = "You are a friendly conversational chatbot."

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=10, memory_key="chat_history", return_messages=True
    )

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Prompt input
if prompt := st.chat_input("Ask a question:"):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Define prompt structure
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    # Create LangChain conversation
    conversation = LLMChain(
        llm=groq_chat,
        prompt=chat_prompt,
        verbose=False,
        memory=st.session_state.memory,
    )

    # Generate response
    response = conversation.predict(human_input=prompt)

    # Append assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
