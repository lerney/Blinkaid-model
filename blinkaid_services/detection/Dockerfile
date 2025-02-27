FROM python:3.11-slim

WORKDIR /app

# Install curl (and optionally clean up to reduce image size)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY ./services/common/requirements.txt /app/common-requirements.txt
COPY ./services/detection/requirements.txt /app/detection-requirements.txt

# Install dependencies from both files
RUN pip install --no-cache-dir -r /app/common-requirements.txt -r /app/detection-requirements.txt