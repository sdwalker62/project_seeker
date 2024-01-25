import toml
import os

from typing import Union
from pathlib import Path
from loguru import logger as log


def convert_option(opt: Union[str, int, bool]) -> str:
    if isinstance(opt, bool):
        if opt:
            return "true"
        else:
            return "false"
    else:
        return opt


if __name__ == "__main__":
    log.info("Starting NITMRE LLM server...")

    conf_file = Path("./config.toml").absolute()
    if conf_file.exists():
        conf = toml.load(str(conf_file))
    else:
        log.critical("Configuration file not found!")
        exit()

    llm_server_conf = conf["llm-server"]

    cmd_str = "./bin/server"
    for opt_k, opt_v in llm_server_conf.items():
        if isinstance(opt_v, bool):
            if opt_v:
                cmd_str += f" --{opt_k}"
        else:
            cmd_str += f" --{opt_k} {opt_v} "
    cmd_str = cmd_str.strip()

    log.info(f"Running command: {cmd_str}")
    os.system(cmd_str)
