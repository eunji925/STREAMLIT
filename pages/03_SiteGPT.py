from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import openai

st.set_page_config(
    page_title="SiteGPT",
    page_icon="💻",
)

st.title("SiteGPT")

def check_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except openai.error.AuthenticationError:
        return False

with st.sidebar:
    docs = None
    keyword = None  
    api_key = st.text_input(
        label="Enter your openAI API-KEY",
        type='password',
    )
    key = False
    if api_key:
        is_valid = check_api_key(api_key)
        if is_valid:
            st.write("Valid OpenAI API Key")
            #key = api_key

        else:
            st.write("Invalid OpenAI API Key")
            st.write("Please Enter Valid API Key")
        
    st.link_button(
        "Github_url",
        "https://github.com/eunji925/STREAMLIT/blob/master/pages/03_SiteGPT.py",
    ) 

if not key:
    st.markdown(
        """
            API key를 입력하고 input box 가 나타나면 Cloudflare 에 대한 질문을 입력하세요.
        """
    )
else:    
    llm = ChatOpenAI(
        openai_api_key=key,
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
    )


    answers_prompt = ChatPromptTemplate.from_template("""
        Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                    
        Then, give a score to the answer between 0 and 5.

        If the answer answers the user question the score should be high, else it should be low.

        Make sure to always include the answer's score even if it's 0.

        Context: {context}
                                                    
        Examples:
                                                    
        Question: How far away is the moon?
        Answer: The moon is 384,400 km away.
        Score: 5
                                                    
        Question: How far away is the sun?
        Answer: I don't know
        Score: 0
                                                    
        Your turn!

        Question: {question}
        """)

    def get_answers(inputs):
        docs = inputs['docs']
        question = inputs['question']
        answers_chain = answers_prompt | llm
        # answers = []
        # for doc in docs:
        #     result = answers_chain.invoke({
        #         "question": question,
        #         "context": doc.page_content
        #     })
        #     answers.append(result.content)
        return {
            "question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {"question": question, "context": doc.page_content}
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata["lastmod"],
                }
                for doc in docs
            ],
        }

    choose_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Use ONLY the following pre-existing answers to answer the user's question.

                Use the answers that have the highest score (more helpful) and favor the most recent ones.

                Cite sources and return the sources of the answers as they are, do not change them.

                Answers: {answers}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    def choose_answer(inputs):
        answers = inputs["answers"]
        question = inputs["question"]
        choose_chain = choose_prompt | llm
        condensed = "\n\n".join(
            f"{answer['answer']}\nSource:{answer['source']}\date:{answer['date']}\n"
            for answer in answers
        )
        return choose_chain.invoke(
            {
                "question": question,
                "answers": condensed,
            }
        )

    def parse_page(soup): # soup : document의 전체 HTML을 가진 beautiful soup object 값
        header = soup.find("header")
        footer = soup.find("footer")
        if header:
            header.decompose()
        if footer:
            footer.decompose()
        return str(soup.get_text()).replace("\n"," ").replace("\t"," ").replace("\xa0", " ") # 공백등을 제거하기 위한 replace
        

    @st.cache_data(show_spinner="Loading website...")
    def load_website(url):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = 800,
            chunk_overlap = 200,
        )
        loader = SitemapLoader(
            url,
            filter_urls=[
            r"^(.*\/workers-ai\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/ai-gateway\/).*",
        ],
            parsing_function = parse_page
        )
        loader.requests_per_second = 1 # 요청 속도 조정 ( 1초에 1번 )
        docs = loader.load_and_split(text_splitter=splitter)
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        return vector_store.as_retriever()


    retriever = load_website("https://developers.cloudflare.com/sitemap-0.xml")
    query = st.text_input("해당 웹사이트에 대해 물어보세요.")
    if query:
        chain = {
            "docs" : retriever,
            "question" : RunnablePassthrough()
        } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
                
        result = chain.invoke(query)
        #st.write(result)
        st.markdown(result.content.replace("\n[출처]", " "))