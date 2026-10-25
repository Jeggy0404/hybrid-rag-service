FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ragkit ragkit
COPY data data
ENV RAG_DB_PATH=/data/rag.db
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "ragkit.api:app", "--host", "0.0.0.0", "--port", "8000"]
