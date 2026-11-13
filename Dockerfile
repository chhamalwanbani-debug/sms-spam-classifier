FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies if any are needed (nltk downloader uses urllib/requests, so we are fine)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time so it's available offline
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy the application files
COPY app.py .
COPY model.pkl .
COPY vectorizer.pkl .

# Expose the port Streamlit runs on (default 8501)
EXPOSE 8501

# Health check to ensure container is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
