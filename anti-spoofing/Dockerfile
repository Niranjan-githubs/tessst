# Use official Python image with CUDA support for TensorFlow
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/main.py .
COPY src/index.html .
COPY models/haarcascade_frontalface_default.xml /app/haarcascade_frontalface_default.xml
COPY models/antispoofing_full_model.h5 /app/antispoofing_full_model.h5

# Expose the application port
EXPOSE 8000

# Use uvicorn for serving
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]