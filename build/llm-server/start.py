import toml
import os
import boto3

from pathlib import Path
from loguru import logger as log

__USE_LOCAL__ = os.environ.get("LOCAL_MODELS", "False") == "True"
__BUCKET_NAME__ = "nitmre-models"
__REMOTE_DIR_NAME__ = "llm/"

def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName) 
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        model_path = os.path.dirname(obj.key).replace('llm', 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        bucket.download_file(obj.key, obj.key.replace('llm', 'models')) # save to same path

if __name__ == "__main__":   
    # Download from S3
    if not __USE_LOCAL__:
        log.info(f"Downloading {__REMOTE_DIR_NAME__} from {__BUCKET_NAME__} S3 Bucket...")
        downloadDirectoryFroms3(__BUCKET_NAME__, __REMOTE_DIR_NAME__)
        log.info(f'{__REMOTE_DIR_NAME__} downloaded from {__BUCKET_NAME__} S3 Bucket...')
    else:
        log.info(f'Skipping S3 download, using local model directory...')

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
