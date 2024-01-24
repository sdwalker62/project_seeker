from langchain_mixtral import Mixtral8x7b
from typing import Union
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import toml
from langchain import hub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path

app = FastAPI()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def llm_response(prompt: str):
    # model = Mixtral8x7b()
    for chunk in rag_chain.stream(prompt):
        # print(chunk)
        # print(chunk.content, end="", flush=True)
        yield chunk


@app.get("/chatbot")
def chatbot(prompt: str):
    return StreamingResponse(llm_response(prompt), media_type="text/event-stream")


def start_server(host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    config_path = Path("./config.toml").absolute()
    if config_path.exists():
        conf = toml.load(str(config_path))
        server_conf = conf["api-server"]
    else:
        raise FileNotFoundError("config.toml not found")

    acronym_path = Path("./acronyms.csv").absolute()
    if acronym_path.exists():
        acronym_path = str(acronym_path)
        csv_loader = CSVLoader(file_path=acronym_path)
        acronym_data = csv_loader.load()
    else:
        raise FileNotFoundError("acronyms.csv not found")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = text_splitter.split_documents(acronym_data)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=HuggingFaceEmbeddings()
    )

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    model = Mixtral8x7b()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    start_server(server_conf["host"], server_conf["port"])
