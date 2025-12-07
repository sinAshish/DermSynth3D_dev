# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set up environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv /opt/venv

WORKDIR /app

# 4. Install system compilers
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 5. Install dependencies (Split steps for reliability)
RUN uv pip install setuptools wheel
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

RUN uv pip install \
    fastapi \
    uvicorn \
    gdown \
    albumentations \
    matplotlib \
    pandas \
    opencv-python-headless \
    websockets \
    trimesh \
    rtree \
    scipy \
    scikit-learn \
    scikit-image \
    numpy

# 6. CRITICAL: Copy the ENTIRE repository root to /app
# Because the Dockerfile is now in the root, this copies 'dermsynth3d' AND 'webapp'
COPY . /app

# 7. Create missing __init__.py files so Python treats folders as packages
# We use 'mkdir -p' to guarantee the path exists before touching files
RUN mkdir -p /app/webapp/api && \
    touch /app/webapp/__init__.py && \
    touch /app/webapp/api/__init__.py

# 8. Set Python Path to include root and submodules
# This ensures 'dermsynth3d' is found regardless of where the script runs
ENV PYTHONPATH="/app:/app/skin3d:$PYTHONPATH"

# 9. Run the application
# We point to the module path inside the 'webapp' package
CMD ["uvicorn", "webapp.api.main:app", "--host", "0.0.0.0", "--port", "8000"]