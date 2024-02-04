import json
import toml
import os

from pprint import pprint
from loguru import logger as log
from typing import Any, Mapping, Iterator, List, Optional
from functools import partial
from requests import request
from pathlib import Path
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration


__LLM_HOST__ = os.environ.get("LLM_SERVER_HOST", "127.0.0.1")
__LLM_PORT__ = os.environ.get("LLM_SERVER_PORT", "7000")
__LLM_API_URL__ = f"http://{__LLM_HOST__}:{__LLM_PORT__}"

log.info("Loading configuration files...")
config_path = Path("./config.toml").absolute()
if config_path.exists():
    conf = toml.load(str(config_path))
    llm_server_conf = conf["llm-server"]
    llm_conf = conf["mixtral8x7b"]

    for k, v in llm_conf.items():
        conf_k = "MIXTRAL8X7B_" + k.upper().replace("-", "_")
        if conf_k in os.environ:
            log.info(f"Overriding {k} with {os.environ[conf_k]}")
            try:
                env_val = float(os.environ[conf_k])
                if env_val.is_integer():
                    log.info(f"Converting {env_val} to int")
                    env_val = int(env_val)
                else:
                    log.info(f"Converting {env_val} to float")
                llm_conf[k] = env_val
            except ValueError as e:
                llm_conf[k] = os.environ[conf_k]

    __MODEL_NAME__ = llm_conf["name"]
    log.debug(f"llm_host: {__LLM_HOST__}")
    log.debug(f"llm_host_port: {__LLM_PORT__}")
    log.debug(f"llm_api_url: {__LLM_API_URL__}")
    log.debug(f"Model name: {__MODEL_NAME__}")
else:
    log.critical("config.toml not found!")
    exit()


__HEADERS__ = {
    "content-type": "application/json",
    "cache-control": "no-cache",
}
__USER_NAME__ = "user"
__PROMPT_HEADER__ = f"This is a conversation between {__MODEL_NAME__} and {__USER_NAME__}, a friendly chatbot. {__MODEL_NAME__} is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.\n\n"
log.debug(f"__PROMPT_HEADER__: {__PROMPT_HEADER__}")

request_dict = llm_conf
# request_dict["prompt"] = __PROMPT_HEADER__


# Partial request for streaming to shorten the code
req = partial(
    request,
    "POST",
    __LLM_API_URL__ + "/completion",
    headers=__HEADERS__,
    stream=False,
)


class Mixtral8x7b(BaseChatModel):
    streaming: bool = False

    def __init__(self, streaming=False, **kwargs: Any) -> None:
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
        prompt: List[HumanMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt_str = prompt[0].content
        request_dict["prompt"] = f"{__USER_NAME__}: {prompt_str}\n{__MODEL_NAME__}: "
        log.info(f"\n\nprompt: {prompt_str}\n\n")
        req_str = json.dumps(request_dict)
        log.debug(f"req_str: {req_str}")
        with req(data=req_str) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    split_val = "data: "
                    res_str = line.split(split_val)[1]
                    res_json = json.loads(res_str)
                    res = res_json["content"]
                    msg = AIMessageChunk(content=res)
                    yield ChatGenerationChunk(message=msg)

        del request_dict["prompt"]

    def _generate(
        self,
        prompt: List[BaseMessage],
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
            # START = prompt[0].content
            # END = prompt[2].content
            # prompt_str = f"{START} {prompt[1].content} {END}"
            request_dict["prompt"] = prompt[0].content + prompt[1].content
            req_str = json.dumps(request_dict)
            print("\n\n\n")
            pprint(request_dict)
            print("\n\n\n")
            resp = req(data=req_str)
            ai_msg = AIMessage(content=resp.json()["content"])
            gens = [ChatGeneration(message=ai_msg)]
            return ChatResult(generations=gens)
