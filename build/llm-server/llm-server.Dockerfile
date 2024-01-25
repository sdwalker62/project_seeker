FROM python:3.9.18

WORKDIR /app
COPY build/llm-server/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY config.toml /app/config.toml
COPY build/llm-server/start.py /app/start.py
CMD ["python", "/app/start.py"]