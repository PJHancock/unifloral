# Use CUDA base image with Python support
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    pkg-config \
    wget \
    git \
    unzip \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv and Python 3.11
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.cargo/bin:/root/.local/bin:${PATH}" && \
    uv python install 3.11
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Create .venv virtual environment with Python 3.11 and install dependencies
RUN uv venv .venv --python 3.11 && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Set MuJoCo rendering environment variables (gymnasium-robotics handles MuJoCo install)
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV DISPLAY=""

# Activate the virtual environment by default
ENV PATH="/app/.venv/bin:${PATH}"

# Copy the rest of the application
COPY . .

# Verify imports
RUN python -c "import gymnasium, minari, gymnasium_robotics; print('✓ All imports successful')"

# Default command
CMD ["/bin/bash"]
