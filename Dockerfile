# Use optimized Python image for faster startup
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install system dependencies for multilingual support
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libicu-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model (lightweight, under 50MB)
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY src/ ./src/

# Create input/output directories with proper permissions
RUN mkdir -p /app/input /app/output && \
    chmod 777 /app/input /app/output

# Set Python path for imports
ENV PYTHONPATH=/app

# Pre-compile Python files for faster startup
RUN python -m compileall src/

# Entry point
CMD ["python", "-O", "src/main.py"]
