import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.schema import SystemMessage
from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun

# 기본 설정
st.set_page_config(page_title="AssistantGPT", page_icon="💼")

st.markdown(
    """
    # AssistantGPT
    Welcome to AssistantGPT. Enter your query and the assistant will research for you using DuckDuckGo or Wikipedia.
    """
)

# 사이드바에서 API 키를 입력받습니다.
with st.sidebar:
    st.markdown(
        """
        [GitHub Repository](https://github.com/paqj/vs-gpt-openai-assistants)
        """
    )
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        st.session_state["api_key"] = api_key

# API 키 확인 후 진행
if "api_key" not in st.session_state:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# 도구 정의
class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearchTool"
    description = """
    Use this tool to perform web searches using DuckDuckGo search engine.
    Example query: 'Latest technology news'
    """
    args_schema = DuckDuckGoSearchToolArgsSchema

    def _run(self, query) -> str:
        try:
            search = DuckDuckGoSearchResults()
            return search.run(query)
        except Exception as e:
            return f"Error during DuckDuckGo search: {str(e)}"


class WikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for on Wikipedia")


class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearchTool"
    description = """
    Use this tool to perform searches on Wikipedia.
    Example query: 'Artificial Intelligence'
    """
    args_schema = WikipediaSearchToolArgsSchema

    def _run(self, query) -> str:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)


class WebScrapingToolArgsSchema(BaseModel):
    url: str = Field(description="The URL of the website you want to scrape")


class WebScrapingTool(BaseTool):
    name = "WebScrapingTool"
    description = """
    Use this to get the content from a link found on DuckDuckGo for research purposes.
    """
    args_schema = WebScrapingToolArgsSchema

    def _run(self, url) -> str:
        loader = WebBaseLoader([url])
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
        return text


class SaveToTXTToolArgsSchema(BaseModel):
    text: str = Field(description="The text you will save to a file.")


class SaveToTXTTool(BaseTool):
    name = "SaveToTXTTOOL"
    description = """
    Save the research result to a .txt file.
    """
    args_schema = SaveToTXTToolArgsSchema

    def _run(self, text) -> str:
        # UTF-8 인코딩으로 파일을 저장하도록 수정
        with open("research_results.txt", "w", encoding="utf-8") as file:
            file.write(text)
        return "Research results saved to research_results.txt"



# OpenAI 어시스턴트 초기화
def setup_openai_agent(api_key):
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4-1106-preview",
        api_key=api_key
    )

    system_message = SystemMessage(
        content="""
            You are a research expert.
            Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 
            When you find a relevant website through DuckDuckGo, scrape the content from that website.
            Ensure the final .txt file contains detailed information, all relevant sources, and citations.
            """
    )

    agent = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=[
            DuckDuckGoSearchTool(),
            WikipediaSearchTool(),
            WebScrapingTool(),
            SaveToTXTTool(),
        ],
        agent_kwargs={"system_message": system_message},
    )

    return agent


# 유저 입력과 어시스턴트 호출
query = st.text_input("Enter the query you want to research")

if query:
    st.write(f"Searching for: {query}")
    
    # OpenAI 어시스턴트 설정
    agent = setup_openai_agent(st.session_state["api_key"])
    
    # 에이전트 실행
    with st.spinner("Researching..."):
        results = agent.run(query)
    
    st.success("Research complete!")
    st.write(results)
    
    # 결과를 파일에 저장
    save_tool = SaveToTXTTool()
    save_tool._run(results)  # 여기서 결과를 파일로 저장합니다.
    
    # 대화 기록 표시
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(f"User: {query}\nAssistant: {results}")
    
    st.markdown("### Conversation History")
    for i, history in enumerate(st.session_state["history"], 1):
        st.markdown(f"**{i}.** {history}")

    # 파일 다운로드 버튼
    with open("research_results.txt", "r", encoding="utf-8") as file:  # encoding="utf-8" 추가
        st.download_button(
            "Download Research Results",
            data=file.read(),
            file_name="research_results.txt",
        )

