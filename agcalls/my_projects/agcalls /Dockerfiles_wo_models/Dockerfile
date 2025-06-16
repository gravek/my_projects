# Этап 1: Установка зависимостей
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt /app/

# Устанавливаем инструменты разработки и зависимости
RUN apt-get update --option Acquire::Retries=3 --option Acquire::http::Timeout=30 && \
    apt-get install -y \
    ffmpeg \
    redis-server \
    supervisor \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Этап 2: Сборка финального образа
FROM python:3.11-slim

WORKDIR /app

# Копируем зависимости из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/bin/redis-server /usr/bin/redis-server
COPY --from=builder /usr/bin/supervisord /usr/bin/supervisord
COPY --from=builder /usr/bin/ffmpeg /usr/bin/ffmpeg

# Копируем папку модели из директории сборки
COPY models--openai--whisper-small /root/.cache/huggingface/hub/models--openai--whisper-small
# COPY models--openai--whisper-small /root/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2
# COPY models--openai--whisper-small /root/.cache/huggingface/hub/
# COPY models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2 /root/.cache/huggingface/hub/

# Устанавливаем runtime-зависимости
RUN apt-get update --option Acquire::Retries=3 --option Acquire::http::Timeout=30 && \
    apt-get install -y \
    ffmpeg \
    redis-server \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы проекта
COPY api_app.py demo_app.py api_server.py /app/
COPY model.onnx /app/
COPY requirements.txt /app/
COPY supervisord.conf /app/
COPY test_api.py /app/

# Устанавливаем переменные окружения для кэша Hugging Face и оффлайн-режима
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0

# Открываем порты
EXPOSE 8000 7860

# Запускаем supervisord
CMD ["supervisord", "-c", "/app/supervisord.conf"]