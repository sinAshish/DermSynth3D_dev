# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set up a virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv /opt/venv

# 4. Use /app as the working directory
WORKDIR /app

# 5. Install system compilers
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 6. STEP 1: Install Build Tools
RUN uv pip install setuptools wheel

# 7. STEP 2: Install PyTorch (CPU Index)
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 8. STEP 3: Install Pytorch3D
RUN uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# 9. STEP 4: Install remaining dependencies
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

# 10. Copy the ENTIRE repository root to /app
COPY . /app

# 11. CRITICAL FIX: Make 'webapp' and 'api' importable packages
# We manually create __init__.py files so Python recognizes the folders as packages
RUN touch /app/webapp/__init__.py /app/webapp/api/__init__.py

# 12. Fix Import Paths
# Add /app (project root) and /app/skin3d to PYTHONPATH
ENV PYTHONPATH="/app:/app/skin3d:$PYTHONPATH"

# 13. Run the App
CMD ["uvicorn", "webapp.api.main:app", "--host", "0.0.0.0", "--port", "8000"]