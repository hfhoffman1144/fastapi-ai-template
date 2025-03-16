# Use NVIDIA CUDA image with Python support
FROM nvidia/cuda:12.0.0-base-ubuntu22.04

# Set up a virtual environment for Python dependencies
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create and activate the virtual environment
RUN python3 -m venv $VIRTUAL_ENV

# Install Python dependencies
WORKDIR /app
COPY ./pyproject.toml /code/pyproject.toml
RUN python -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    pip install --no-cache-dir /code/.

# Copy application code
COPY ./src /app

# Expose port for FastAPI
EXPOSE 8000

# Ensure NVIDIA GPU is accessible
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]