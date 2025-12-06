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

# 5. STEP 1: Install Torch AND Build Tools
# We MUST install 'setuptools' and 'wheel' here because --no-build-isolation
# relies on them being present in the environment.
RUN uv pip install \
    torch \
    torchvision \
    setuptools \
    wheel \
    --index-url https://download.pytorch.org/whl/cpu

# 6. STEP 2: Install Pytorch3D
# Now this works because 'torch' AND 'setuptools' are already installed.
RUN uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# 7. STEP 3: Install the rest of your stack
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

# 8. Copy your app code
COPY . .

# 9. Run the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]