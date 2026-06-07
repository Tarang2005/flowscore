FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (Render uses $PORT, HuggingFace uses 7860, Koyeb uses 8000)
# We will read PORT from environment or default to 8000
ENV PORT=8000
EXPOSE $PORT

# Run the FastAPI application
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
