FROM nvcr.io/nvidia/cuda:11.7.1-runtime-ubuntu22.04
# FROM alpine:3.19.1
LABEL maintainer="Athena ML"
LABEL version="1.0"

# COPY --from=nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04 /usr/local/cuda/ /usr/local/cuda/

COPY --from=continuumio/miniconda3:latest /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app/

RUN conda install -c conda-forge -y python=3.10

# COPY models/ /app/models/
COPY /build/llm-server/bin /app/bin/
COPY build/llm-server/start.py /app/start.py
COPY config.toml /app/config.toml
COPY build/llm-server/requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

ENV LLM_SERVER_THREADS=128
ENV LLM_SERVER_THREADS_BATCH=128

CMD ["python3", "/app/start.py"]
