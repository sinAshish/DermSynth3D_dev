# 1. Use Full Python 3.10 (Fixes many missing system lib issues)
FROM python:3.10

# 2. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set up environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv /opt/venv

# 4. Set Workdir to Root initially
WORKDIR /app

# 5. Install ALL system libraries required by OpenCV and Pytorch3D
# We added libsm6, libxext6, libxrender-dev which are often needed by opencv-headless
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 6. Install Python Dependencies (Split for safety)
RUN uv pip install setuptools wheel

# Install Torch CPU
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CRITICAL: Explicitly install Pytorch3D dependencies first
RUN uv pip install fvcore iopath portalocker

# Install Pytorch3D
RUN uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# Install App Dependencies
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
# Ensure Render "Root Directory" is set to "." so this copies everything (webapp + dermsynth3d)
COPY . /app

# 8. Create __init__.py files
RUN mkdir -p /app/webapp/api && \
    touch /app/webapp/__init__.py && \
    touch /app/webapp/api/__init__.py

# 9. Set PYTHONPATH to find 'dermsynth3d'
ENV PYTHONPATH="/app:/app/skin3d:$PYTHONPATH"

# 10. Switch to webapp directory
WORKDIR /app/webapp

# 11. DEBUG START COMMAND
# This runs a python script to import your app *before* uvicorn.
# If there is a missing library, this will print the FULL STACK TRACE to the logs.
CMD ["sh", "-c", "python -c 'from api import main; print(\"Import Successful!\")' && uvicorn api.main:app --host 0.0.0.0 --port 10000"]