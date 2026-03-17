FROM python:3.9-slim
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --default-timeout=10000 -r requirements.txt

CMD ["python", "application.py"]