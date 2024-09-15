import streamlit as st
import openai
from pydantic import BaseModel, Field

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
    st.link_button(
        "Github_url",
        "https://github.com/eunji925/STREAMLIT/blob/master/pages/04_Assistant.py",
    )
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        st.session_state["api_key"] = api_key

# API 키 확인 후 진행
if "api_key" not in st.session_state:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()


class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class WikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for on Wikipedia")


class WebScrapingToolArgsSchema(BaseModel):
    url: str = Field(description="The URL of the website you want to scrape")


class SaveToTXTToolArgsSchema(BaseModel):
    text: str = Field(description="The text you will save to a file.")


def perform_search(query, api_key):
    system_message = {
        "role": "system",
        "content": """
        You are a research expert. 
        Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 
        When you find a relevant website through DuckDuckGo, scrape the content from that website. 
        Ensure the final result contains detailed information, all relevant sources, and citations.
        """
    }

    user_message = {"role": "user", "content": query}

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[system_message, user_message],
        api_key=api_key
    )

    return response['choices'][0]['message']['content']


def save_results_to_file(text):
    with open("research_results.txt", "w", encoding="utf-8") as file:
        file.write(text)
    return "Research results saved to research_results.txt"


# 유저 입력과 어시스턴트 호출
query = st.text_input("Enter the query you want to research")

if query:
    st.write(f"Searching for: {query}")

    # OpenAI 어시스턴트 호출
    with st.spinner("Researching..."):
        try:
            results = perform_search(query, st.session_state["api_key"])
            st.success("Research complete!")
            st.write(results)
            
            # 결과를 파일에 저장
            save_results_to_file(results)
        except Exception as e:
            st.error(f"An error occurred: {e}")

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
