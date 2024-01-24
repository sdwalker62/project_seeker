import uvicorn
import toml

from langchain_mixtral import Mixtral8x7b
from typing import Union
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger as log
from langchain import hub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path

app = FastAPI()


def format_docs(docs) -> None:
    return "\n\n".join(doc.page_content for doc in docs)


def llm_response(prompt: str) -> str:
    log.info("Starting LLM response...")
    for chunk in rag_chain.stream(prompt):
        yield chunk


@app.get("/chatbot")
def chatbot(prompt: str) -> StreamingResponse:
    return StreamingResponse(llm_response(prompt), media_type="text/event-stream")


def start_server(host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    log.info("Starting server...")
    log.info("Loading configuration files...")
    config_path = Path("./config.toml").absolute()
    if config_path.exists():
        conf = toml.load(str(config_path))
        server_conf = conf["api-server"]
    else:
        log.critical("config.toml not found!")
        exit()

    log.info("Looking for acronym csv...")
    acronym_path = Path("./acronyms.csv").absolute()
    if acronym_path.exists():
        log.info("found it!")
        log.info("Loading acronym data...")
        acronym_path = str(acronym_path)
        csv_loader = CSVLoader(file_path=acronym_path)
        acronym_data = csv_loader.load()
        log.info("Loaded acronym data!")
    else:
        log.critical("acronyms.csv not found!")
        exit()

    log.info("Loading vectorstore...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = text_splitter.split_documents(acronym_data)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=HuggingFaceEmbeddings()
    )
    log.info("Loaded vectorstore!")

    log.info("Loading retriever...")
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    log.info("Loaded retriever!")

    log.info("Loading model...")
    model = Mixtral8x7b()
    log.info("Loaded model!")

    log.info("Creating 'RAG' chain...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    # rag_chain_with_source = RunnableParallel(
    #     {"context": retriever, "question": RunnablePassthrough()}
    # ).assign(answer=rag_chain)
    log.info("Created 'RAG' chain!")

    start_server(server_conf["host"], server_conf["port"])
