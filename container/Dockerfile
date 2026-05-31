FROM python:3.11-slim

WORKDIR /app

COPY container/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY container/app /app/app
COPY container/models /app/models
COPY container/STUDENT.json /app/STUDENT.json

ENTRYPOINT ["python", "/app/app/cli.py"]
