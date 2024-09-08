import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import json
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"././.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100, 
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_resource(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

output_parser = JsonOutputParser()
questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an assistant in the role of a teacher. Give 15 problems based on the received context. Each problem has 4 options. Only one of the choices is correct. Mark the correct answer using (o).Please refer to the example below. The difficulty levels of the questions are Hard and Easy. Set it randomly. And please specify the difficulty level next to the problem.
         
                Question examples:                    
                    Question: What is the color of the ocean? (Hard)
                    Answers: Red|Yellow|Green|Blue(o)
                        
                    Question: What is the capital or Georgia? (Easy)
                    Answers: Baku|Tbilisi(o)|Manila|Beirut
                        
                    Question: When was Avatar released? (Easy)
                    Answers: 2007|2001|2009(o)|1998
                        
                    Question: Who was Julius Caesar? (Hard)
                    Answers: A Roman Emperor(o)|Painter|Actor|Model
                                    
                Context: {context}
            """
        )
    ]
)
formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a powerful formatting algorithm.
                
                You format exam questions into JSON format.
                Answers with (o) are the correct ones.
                
                Example Input:
                    Question: What is the color of the ocean? (Hard)
                    Answers: Red|Yellow|Green|Blue(o)
                        
                    Question: What is the capital or Georgia? (Easy)
                    Answers: Baku|Tbilisi(o)|Manila|Beirut
                        
                    Question: When was Avatar released? (Easy)
                    Answers: 2007|2001|2009(o)|1998
                        
                    Question: Who was Julius Caesar? (Hard)
                    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
                Example Output:                
                    ```json
                    {{ "questions": [
                            {{
                                "question": "What is the color of the ocean? (Hard)",
                                "answers": [
                                    {{
                                        "answer": "Red",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Yellow",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Green",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Blue",
                                        "correct": true
                                    }},
                                ],
                                "level": "Hard"
                            }},
                            {{
                                "question": "What is the capital or Georgia? (Easy)",
                                "answers": [
                                    {{
                                        "answer": "Baku",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Tbilisi",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "Manila",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Beirut",
                                        "correct": false
                                    }},
                                ],
                                "level": "Easy"
                            }},
                            {{
                                "question": "When was Avatar released? (Easy)",
                                "answers": [
                                    {{
                                        "answer": "2007",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2001",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2009",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "1998",
                                        "correct": false
                                    }},
                                ],
                                "level": "Easy"
                            }},
                            {{
                                "question": "Who was Julius Caesar? (Hard)",
                                "answers": [
                                    {{
                                        "answer": "A Roman Emperor",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "Painter",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Actor",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Model",
                                        "correct": false
                                    }},
                                ],
                                "level": "Hard"
                            }}
                        ]
                    }}
                    ```
                Questions: {context}
            """,
        )
    ]
)

with st.sidebar:
    docs = None
    keyword = None  
    api_key = st.text_input(
        label="Enter your openAI API-KEY",
        type='password',
    )
    if api_key:
        level = st.selectbox(
            label="Choice Levels",
            options=("Hard", "Easy")
        )

        choice = st.selectbox(
            label="Choice Options",
            options=("File", "Wikipedia")
        )
        
        if choice == "File":  
            file = st.file_uploader(
                label="Upload File...",
                type=['txt']
            )
            if file:
                docs = split_file(file) 
        else: 
            keyword = st.text_input(
                label='Enter the keyword you want to search'
            )
            if keyword:
                docs = wiki_search(keyword)  

    st.link_button(
        "Github_url",
        "https://github.com/eunji925/STREAMLIT/blob/c5b9d0ab07343d52df18e2e6d75a1d2450e60af1/pages/02_QuizGPT.py",
    )       

if not docs:
    st.markdown(
    """
        We will create a quiz related to the document you want.
                
        Upload a file or enter a topic.
    """
    )
else:
    llm = ChatOpenAI(
        api_key=api_key,
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    
    questions_chain = {"context": format_docs} | questions_prompt | llm
    formatting_chain = formatting_prompt | llm

    response = run_quiz_chain(docs, keyword if keyword else file.name)  


    questions = [question for question in response["questions"] if question['level'] == level]
    correct_count = 0  
    total_questions = len(questions)  

    with st.form("questions_form"):        
        for i, question in enumerate(questions):
            st.write(f"**{i+1}. {question['question']}**")
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"q_{i}"
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                correct_count += 1
            elif value is not None:
                st.error("Wrong!")
                    
        button = st.form_submit_button()
        if button:            
            if correct_count == total_questions:  
                st.write("모두 정답입니다.")
            else:  
                st.warning(f"{total_questions} 중 {correct_count} 개가 정답입니다.")
