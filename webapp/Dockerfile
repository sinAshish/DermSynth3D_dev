# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set up environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv /opt/venv

# 4. Set WORKDIR to Root first to copy everything
WORKDIR /app

# 5. Install system compilers
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 6. Install dependencies
# Note: We assume requirements files are copied with the rest of the repo below,
# OR you can copy them explicitly here if you prefer.
# For now, we install packages directly to be safe.
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

# 7. COPY THE ROOT REPO
# This copies 'dermsynth3d', 'webapp', etc. into /app
COPY . /app

# 8. Create __init__.py files
# Ensure 'api' is treated as a package
RUN mkdir -p /app/webapp/api && \
    touch /app/webapp/api/__init__.py

# 9. Set PYTHONPATH
# We add /app so that 'dermsynth3d' (located at /app/dermsynth3d) can be found
ENV PYTHONPATH="/app:/app/skin3d:$PYTHONPATH"

# 10. CRITICAL CHANGE: Switch directory to 'webapp'
# This replicates your local setup.
WORKDIR /app/webapp

# 11. Run the application
# Now we use 'api.main:app' because we are inside the webapp folder
CMD ["uvicorn", "api.main:app", "--reload", "--host", "0.0.0.0", "--port", "10000"]