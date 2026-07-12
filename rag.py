import os
from dotenv import load_dotenv
from utils import load_options
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_localai import LocalAIEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence

load_dotenv()
options = load_options("options.yaml")


llm = init_chat_model(
    model=options["model"],
    model_provider="openai",
    base_url=options["base_url"],
    api_key=os.getenv("LLM_API_KEY")
)

embedding_model = LocalAIEmbeddings(
    openai_api_base=options["base_url"],
    openai_api_key=os.getenv("LLM_API_KEY"),
    model=options["embedding_model"],
)

db = Chroma(
    persist_directory=options["chroma_dir"],
    embedding_function=embedding_model
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}
)

def reformulate_question():
    reformulation_prompt = """
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", reformulation_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=chat_template
    )

    return history_aware_retriever


def answer_question():
    answer_question_prompt = """ 
    Use the following pieces of retrieved context to answer the question. \
    Use three to seven sentences maximum and keep the answer concise, while still giving depth.\
    Structure your output in a list. \

    {context}"""

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", answer_question_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ]
    )

    answer_question_chain = create_stuff_documents_chain(
        llm=llm, 
        prompt=chat_template
    )

    history_aware_retriever = reformulate_question()
    rag_chain = create_retrieval_chain(history_aware_retriever, answer_question_chain)
    return rag_chain


class ChatState(TypedDict):
    input: str
    history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


def call_model(state: ChatState):
    rag_chain = answer_question()
    response = rag_chain.invoke(state)

    return {
        "history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"])
        ],
        "context": response["context"],
        "answer": response["answer"]
    }

def make_app():
    workflow = StateGraph(state_schema=ChatState)
    workflow.add_edge(START, 'model')
    workflow.add_node('model', call_model)

    memory = MemorySaver()

    app = workflow.compile(checkpointer=memory)
    return app

if __name__ == "__main__":
    app = make_app()
    result = app.invoke(
        {"input": input("Ask something: ")},
        config={"configurable": {"thread_id": "123asd"}}
    )

    print(result["answer"])