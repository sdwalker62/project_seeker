import toml
import os

from pathlib import Path
from loguru import logger as log

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
        os_key = "LLM_SERVER_" + opt_k.upper().replace("-", "_")
        if os_key in os.environ:
            opt_v = os.environ[os_key]

        if isinstance(opt_v, bool):
            if opt_v:
                cmd_str += f" --{opt_k}"
        else:
            cmd_str += f" --{opt_k} {opt_v} "
    cmd_str = cmd_str.strip()

    log.info(f"Running command: {cmd_str}")
    os.system(cmd_str)
