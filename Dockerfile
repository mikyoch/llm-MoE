FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e ".[server]"

EXPOSE 8000

CMD ["unified-llm", "serve", "--config", "examples/config.yaml", "--host", "0.0.0.0", "--port", "8000"]
