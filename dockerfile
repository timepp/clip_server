FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
COPY clip_server.py .
EXPOSE 5000
CMD ["python", "clip_server.py"]
