from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import openai
import streamlit as st
import openai
import os

st.set_page_config(
    page_title="APP",
    page_icon="💕",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_path = f"./.cache/files/{file.name}"
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_content = file.read()
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    # OpenAIEmbeddings애 api_key를 전달하였습니다.
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def check_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except openai.error.AuthenticationError:
        return False

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("APP_document")

st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!

    Upload your files and API key on the sidebar.
"""
)

with st.sidebar:
    API_KEY = st.text_input("Please Enter Your OpenAI API Key", type="password")
    is_file = False
    if API_KEY:
        is_valid = check_api_key(API_KEY)
        if is_valid:
            st.write("Valid OpenAI API Key")
            file = st.file_uploader(
                "Upload a .txt .pdf or .docx file",
                type=["pdf", "txt", "docx"],
                disabled=not is_valid,
            )
            if file:
                is_file = True
        else:
            st.write("Invalid OpenAI API Key")
            st.write("Please Enter Valid API Key")
    else:
        is_valid = False

    st.link_button(
        "Github_url",
        "https://github.com/eunji925/STREAMLIT/blob/master/app.py",
    )

    
    st.markdown(
            """
            from langchain.prompts import ChatPromptTemplate
            from langchain.document_loaders import UnstructuredFileLoader
            from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
            from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
            from langchain.storage import LocalFileStore
            from langchain.text_splitter import CharacterTextSplitter
            from langchain.vectorstores.faiss import FAISS
            from langchain.chat_models import ChatOpenAI
            from langchain.callbacks.base import BaseCallbackHandler
            import openai.error
            import streamlit as st
            import openai
            import os
            st.set_page_config(
                page_title="APP",
                page_icon="💕",
            )
            class ChatCallbackHandler(BaseCallbackHandler):
                message = ""
                def on_llm_start(self, *args, **kwargs):
                    self.message_box = st.empty()
                def on_llm_end(self, *args, **kwargs):
                    save_message(self.message, "ai")
                def on_llm_new_token(self, token, *args, **kwargs):
                    self.message += token
                    self.message_box.markdown(self.message)
            @st.cache_data(show_spinner="Embedding file...")
            def embed_file(file):
                file_path = f"./.cache/files/{file.name}"
                folder_path = os.path.dirname(file_path)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_content = file.read()
                with open(file_path, "wb") as f:
                    f.write(file_content)
                cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
                splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    separator="\n",
                    chunk_size=600,
                    chunk_overlap=100,
                )
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load_and_split(text_splitter=splitter)
                # OpenAIEmbeddings애 api_key를 전달하였습니다.
                embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
                cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
                vectorstore = FAISS.from_documents(docs, cached_embeddings)
                retriever = vectorstore.as_retriever()
                return retriever
            def save_message(message, role):
                st.session_state["messages"].append({"message": message, "role": role})
            def send_message(message, role, save=True):
                with st.chat_message(role):
                    st.markdown(message)
                if save:
                    save_message(message, role)
            def paint_history():
                for message in st.session_state["messages"]:
                    send_message(
                        message["message"],
                        message["role"],
                        save=False,
                    )
            def format_docs(docs):
                return "\n\n".join(document.page_content for document in docs)
            def check_api_key(api_key):
                try:
                    openai.api_key = api_key
                    openai.Model.list()
                    return True
                except openai.error.AuthenticationError:
                    return False
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "
                        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                        Context: {context}
                        ",
                    ),
                    ("human", "{question}"),
                ]
            )
            st.title("APP_document")
            st.markdown(
                "
                Welcome!
                Use this chatbot to ask questions to an AI about your files!
                Upload your files and API key on the sidebar.
            "
            )
            with st.sidebar:
                API_KEY = st.text_input("Please Enter Your OpenAI API Key", type="password")
                is_file = False
                if API_KEY:
                    is_valid = check_api_key(API_KEY)
                    if is_valid:
                        st.write("Valid OpenAI API Key")
                        file = st.file_uploader(
                            "Upload a .txt .pdf or .docx file",
                            type=["pdf", "txt", "docx"],
                            disabled=not is_valid,
                        )
                        if file:
                            is_file = True
                    else:
                        st.write("Invalid OpenAI API Key")
                        st.write("Please Enter Valid API Key")
                else:
                    is_valid = False
                st.link_button(
                    "Go to Github Repo",
                    "https://github.com/eunji925/STREAMLIT/blob/master/app.py",
                )
            if is_file:
                retriever = embed_file(file)
                send_message("I'm ready! Ask away!", "ai", save=False)
                paint_history()
                message = st.chat_input("Ask anything about your file...")
                if message:
                    send_message(message, "human")
                    chain = (
                        {
                            "context": retriever | RunnableLambda(format_docs),
                            "question": RunnablePassthrough(),
                        }
                        | prompt
                        | ChatOpenAI(
                            temperature=0.1,
                            streaming=True,
                            callbacks=[
                                ChatCallbackHandler(),
                            ],
                            openai_api_key=API_KEY,
                        )
                    )
                    with st.chat_message("ai"):
                        chain.invoke(message)
            else:
                st.session_state["messages"] = []
            """
        )

if is_file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | ChatOpenAI(
                temperature=0.1,
                streaming=True,
                callbacks=[
                    ChatCallbackHandler(),
                ],
                openai_api_key=API_KEY,
            )
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []