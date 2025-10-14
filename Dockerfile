FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.8.23 /uv /uvx /bin/
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
# Copy the repository
COPY . /repo

# Set working directory
WORKDIR /repo
SHELL ["/bin/bash", "-c"]

# Create virtual environment
RUN uv venv --python 3.12 .venv

# Activate environment and install dependencies
ENV PATH="/repo/.venv/bin:$PATH"
RUN uv pip install -e ".[dev, test]"

# The container is now ready for running tests or other commands