FROM python:3.10-slim

# System dependencies for audio processing and GPU support
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose any necessary ports (if needed for future development)
EXPOSE 8000

# Default command
CMD ["bash"]