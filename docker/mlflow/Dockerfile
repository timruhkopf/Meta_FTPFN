#FROM python:3.11
#
#COPY requirements.txt /tmp/
#RUN pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.11-slim

# Install system dependencies for Postgres
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt