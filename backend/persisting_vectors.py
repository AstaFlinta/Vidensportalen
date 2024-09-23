import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from langchain_community.document_loaders import JSONLoader
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage


load_dotenv()

def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["url"] = record.get("url")
    metadata["title"] = record.get("title")

    return metadata

loader = JSONLoader(
    file_path="./copy_scraped_data.json",
    jq_schema=".[]",
    content_key="text",
    metadata_func=metadata_func
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory="vector_storage2"
    )

vectorstore.persist()
