from typing import Any, Mapping, AsyncIterator, Dict, Iterator, List, Optional, cast
from functools import partial
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from requests import request
import json
import ast

from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompt_values import PromptValue

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

request_dict = dict()
request_dict["stream"] = True
request_dict["n_predict"] = 400
request_dict["temperature"] = 0.7
request_dict["stop"] = ["</s>", f"{__MODEL_NAME__}:", f"{__USER_NAME__}:"]
request_dict["repeat_last_n"] = 256
request_dict["repeat_penalty"] = 1.18
request_dict["top_k"] = 40
request_dict["top_p"] = 0.95
request_dict["min_p"] = 0.05
request_dict["tfs_z"] = 1
request_dict["typical_p"] = 1
request_dict["presence_penalty"] = 0
request_dict["frequency_penalty"] = 0
request_dict["mirostat"] = 0
request_dict["mirostat_tau"] = 5
request_dict["mirostat_eta"] = 0.1
request_dict["grammar"] = ""
request_dict["n_probs"] = 0
request_dict["image_data"] = []
request_dict["cache_prompt"] = True
request_dict["api_key"] = ""
request_dict["slot_id"] = 0
request_dict["prompt"] = __PROMPT_HEADER__


# Partial request for streaming to shorten the code
req = partial(
    request,
    "POST",
    __LLM_API_URL__ + "/completion",
    headers=__HEADERS__,
    stream=True,
)


class Mixtral8x7b(BaseChatModel):
    streaming: bool = True

    def __init__(self, streaming=True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.streaming = streaming

    @property
    def _llm_type(self) -> str:
        return "mixtral8x7b"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "streaming": self.streaming,
        }

    def convert_msg_to_prompt(self, msg: BaseMessage) -> str:
        if isinstance(msg, HumanMessage):
            return f"{__USER_NAME__}: {msg.text}"
        elif isinstance(msg, SystemMessage):
            return f"{__MODEL_NAME__}: {msg.text}"
        else:
            raise NotImplementedError

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "mixtral8x7b"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request_dict["prompt"] += f"{__USER_NAME__}: {prompt}\n{__MODEL_NAME__}: "
        req_str = json.dumps(request_dict)
        with req(data=req_str) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    split_val = "data: "
                    res_str = line.split(split_val)[1]
                    res_json = json.loads(res_str)
                    res = res_json["content"]
                    msg = AIMessageChunk(content=res)
                    yield ChatGenerationChunk(message=msg)

    def _generate(
        self,
        prompt: BaseMessage,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            return generate_from_stream(stream_iter)
        else:
            raise NotImplementedError
