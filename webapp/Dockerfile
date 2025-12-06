# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set up a virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv /opt/venv

WORKDIR /webapp

# 4. Install system compilers (Required for Pytorch3D)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 5. STEP 1: Install Build Tools (From standard PyPI)
# We install these first so they are available for the next steps.
RUN uv pip install setuptools wheel

# 6. STEP 2: Install PyTorch (From PyTorch CPU Repo)
# We do this separately so we can target the CPU index without breaking other packages
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 7. STEP 3: Install Pytorch3D
# We use --no-build-isolation so it uses the 'torch' and 'setuptools' we just installed
RUN uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# 8. STEP 4: Install the rest of your stack (From standard PyPI)
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

# 9. Copy your app code
COPY . .

# 10. Run the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]