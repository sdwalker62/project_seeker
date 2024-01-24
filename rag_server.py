from langchain_mixtral import Mixtral8x7b
from typing import Union
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import requests
import json
import ast
import tomllib

app = FastAPI()


def llm_response(prompt: str):
    model = Mixtral8x7b()
    for chunk in model.stream(prompt):
        print(chunk.content, end="", flush=True)
        yield chunk.content


@app.get("/chatbot")
def chatbot(prompt: str):
    return StreamingResponse(llm_response(prompt), media_type="text/event-stream")


def start_server(host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        conf = tomllib.load(f)
        server_conf = conf["api-server"]

    start_server(server_conf["host"], server_conf["port"])
