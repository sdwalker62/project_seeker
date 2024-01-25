FROM nvcr.io/nvidia/cuda:11.7.1-runtime-ubuntu22.04
COPY --from=continuumio/miniconda3:latest /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app/
COPY build/llm-server/requirements.txt /app/requirements.txt
RUN conda install -c conda-forge -y python=3.10
RUN pip3 install -r /app/requirements.txt

COPY config.toml /app/config.toml
COPY build/llm-server/start.py /app/start.py
CMD ["python3", "/app/start.py"]
