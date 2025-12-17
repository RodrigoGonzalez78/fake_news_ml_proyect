# Fake News Detector - Web Interface (CPU Only)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies for TensorFlow CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Create data directory for feedback
RUN mkdir -p /app/data/feedback

# Expose FastHTML default port
EXPOSE 5001

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the web application
CMD ["python", "-m", "uvicorn", "src.web.main:app", "--host", "0.0.0.0", "--port", "5001"]
