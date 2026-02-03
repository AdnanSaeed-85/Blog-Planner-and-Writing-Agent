from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.graph.message import add_messages
from CONFIG import OPENAI_MODEL, GROQ_MODEL
from langchain_groq import ChatGroq

load_dotenv()

# llm = ChatOpenAI(model=OPENAI_MODEL)
llm = ChatGroq(model=GROQ_MODEL)

class state_class(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class task(BaseModel):
    id: int
    title: str = Field(description="add title of the topic")
    brief: str = Field(description='add brielf discussion about this particular topic')
class plan(BaseModel):
    blog_str: str = Field(description="add text of blog")
    tasks: List[task] = Field(description="add list of tasks", default_factory=list)

