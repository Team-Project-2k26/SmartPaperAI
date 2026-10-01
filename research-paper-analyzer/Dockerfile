FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for PyMuPDF and spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy project source (order: deps-heavy dirs last for layer caching)
COPY document_processing ./document_processing
COPY summarization ./summarization
COPY qa_model ./qa_model
COPY backend ./backend
COPY uploads ./uploads

# Create uploads dir
RUN mkdir -p uploads

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
