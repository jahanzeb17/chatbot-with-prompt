

import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import docx
import io

from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
import chromadb

load_dotenv()

# llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
llm = ChatGroq(model="llama3-70b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


PERSIST_DIRECTORY = "./chroma_langchain_db"

def init_session_state():
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def load_data(files):
    all_documents = []

    if not files:
        return all_documents

    for uploaded_file in files:
        try:
            # Read text content from UploadedFile
            text = uploaded_file.getvalue().decode("utf-8")  # Decode bytes to string
            document = Document(page_content=text, metadata={"source": uploaded_file.name})
            all_documents.append(document)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    return all_documents

    


def chunk_data(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    texts = [Document(page_content=chunk.page_content) for chunk in chunks]
    return texts


def vectordb(texts):

    text_list = [doc.page_content for doc in texts]
    vectordb = FAISS.from_texts(text_list, embeddings)  
    return vectordb.as_retriever() 


template = """
    You are an AI that mimics the personality and communication style of a specific user. 
    Use the following context to answer the user's query. Maintain the tone and style that is within the context.
    
    Context:
    {context}

    Chat History:
    {chat_history}

    Few-shot Examples:
    {fewshot_examples}

    User Query: {query}

    Answer:
"""


def format_fewshot_examples():

    fewshot_examples = [
    {"input": "Hi, Raj. How was your weekend?", "output": "Hey, Anjali. My weekend was great. I watched a great movie."},
    {"input": "Oh really? What was the name of the movie you watched?", "output": "I watched Avengers Endgame. It is the last movie of the Avengers."},
    {"input": "Oh, I have watched Avengers Endgame too. I loved the movie.", "output": "Really? Who is your favourite Avenger?"},
    {"input": "I can’t name one! Iron Man, Thor, Captain America, Captain Marvel, Scarlet Witch and Black Widow, to name a few.", "output": "Wow, you have some of the strongest Avengers there! I have the same choice except that I loved Spider-Man too."},
    {"input": "My sister took me to see the movie as soon as it was released. Both me and my sister have been great fans of Avengers since childhood.", "output": "Oh wow! I am myself a big fan of Avengers and have watched all the movies. I too wanted to go to the theatre and watch the movie, but I was out of station for a family function."},
    {"input": "Oh I see. The movie stood up to all the expectations that the audience had after watching the trailer. In fact, I would say the movie surpassed expectations.", "output": "Very true. There was no better way to finish the Avengers, I believe. The movie just took me through a rollercoaster of emotions."},
    {"input": "True! Just when I was feeling happy that the Avengers got rid of Thanos for good, the next moment I was bawling my eyes out seeing Iron Man had sacrificed himself to save the world and everyone else.", "output": "We can’t ever see Black Widow, Iron Man and Captain America ever in any Marvel movies."},
    {"input": "Yes, very sad. Anyway, it was nice talking to you. See you tomorrow in school. Bye.", "output": "Same here. Bye."}
]

    return "\n".join([f"User: {ex['input']}\nAI: {ex['output']}" for ex in fewshot_examples])


def get_llm_response(retriever, query):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in st.session_state.chat_history])
    formatted_fewshots = format_fewshot_examples()

    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(context=context, chat_history=chat_history_str, fewshot_examples=formatted_fewshots, query=query)
    
    response = llm.invoke(messages).content
    return response


def main():
    st.title("Personality Modeling with Fewshot Prompt")
    init_session_state()

    for message in st.session_state.chat_history:
        with st.chat_message("human" if isinstance(message, HumanMessage) else "ai"):
            st.markdown(message.content)

    with st.sidebar:
        st.header("Upload Data Here")
        uploaded_files = st.file_uploader(
            "Upload documents", 
            type=["pdf", "txt", "doc", "docx"],
            accept_multiple_files=True
        )
        button = st.button("Process")

        if button:
            with st.spinner("Processing..."):
                loaded_docs = load_data(uploaded_files)
                chunks = chunk_data(loaded_docs)
                st.session_state.retriever = vectordb(chunks)
                st.success("Vector DB Updated!")

    query = st.chat_input("Enter your query:")
    
    if query:
        with st.chat_message("human"):  
            st.markdown(query)

        st.session_state.chat_history.append(HumanMessage(content=query))

        if st.session_state.retriever:
            response = get_llm_response(st.session_state.retriever, query)

            with st.chat_message("ai"):  
                st.markdown(response)

            st.session_state.chat_history.append(AIMessage(content=response))

if __name__=="__main__":
    main()