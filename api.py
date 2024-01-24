from typing import Union
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import requests
import json
import ast

__HEADERS__ = {
    "content-type": "application/json",
    "cache-control": "no-cache",
}
__LLM_API_URL__ = "http://localhost:8080"
__SERVER_URL__ = "127.0.0.1"
__SERVER_PORT__ = 8000

__MODEL_NAME__ = "Mixtral"
__USER_NAME__ = "NITMRE_USER"
__PROMPT_HEADER__ = f"This is a conversation between {__USER_NAME__} and {__MODEL_NAME__}, a friendly chatbot. {__MODEL_NAME__} is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.\n\n"


app = FastAPI()

request = dict()
request["stream"] = True
request["n_predict"] = 400
request["temperature"] = 0.7
request["stop"] = ["</s>", f"{__MODEL_NAME__}:", f"{__USER_NAME__}:"]
request["repeat_last_n"] = 256
request["repeat_penalty"] = 1.18
request["top_k"] = 40
request["top_p"] = 0.95
request["min_p"] = 0.05
request["tfs_z"] = 1
request["typical_p"] = 1
request["presence_penalty"] = 0
request["frequency_penalty"] = 0
request["mirostat"] = 0
request["mirostat_tau"] = 5
request["mirostat_eta"] = 0.1
request["grammar"] = ""
request["n_probs"] = 0
request["image_data"] = []
request["cache_prompt"] = True
request["api_key"] = ""
request["slot_id"] = 0
request["prompt"] = __PROMPT_HEADER__

counter = 0


async def llm_response(prompt: str):
    request["prompt"] += f"{__USER_NAME__}: {prompt}\n{__MODEL_NAME__}: "
    with requests.request(
        "POST",
        __LLM_API_URL__ + "/completion",
        data=json.dumps(request),
        headers=__HEADERS__,
        stream=True,
    ) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                res_str = line.split("data: ")[1]
                res_json = json.loads(res_str)
                res = res_json["content"]
                yield res


@app.get("/chatbot")
def chatbot(prompt: str):
    return StreamingResponse(llm_response(prompt), media_type="text/event-stream")


@app.get("/health_check")
async def health_check():
    health = requests.get(__LLM_API_URL__ + "/health")
    return health.json()


def start_server() -> None:
    uvicorn.run(
        app,
        host=__SERVER_URL__,
        port=__SERVER_PORT__,
    )


if __name__ == "__main__":
    start_server()
