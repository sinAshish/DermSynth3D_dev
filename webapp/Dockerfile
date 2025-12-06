# 1. Use Python 3.10 (Crucial for Pytorch3D compatibility)
FROM python:3.10-slim

# 2. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set up a virtual environment and put it in the PATH
# This ensures all subsequent 'uv pip install' commands land in the same place
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv /opt/venv

WORKDIR /webapp

# 4. Install system compilers (Git + GCC + G++)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 5. STEP 1: Install PyTorch (CPU version) explicitly FIRST
# We use the CPU version to keep the image size small for Render
# We DO NOT install pytorch3d yet.
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. STEP 2: Install Pytorch3D with --no-build-isolation
# This tells uv: "Don't create a clean env. Use the environment where I just installed Torch."
RUN uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# 7. STEP 3: Install the rest of your stack
# Note: We use 'uv pip install' instead of 'uv add'
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

# 9. (Optional) Handle your data download
# If you have the tar file locally, copy it. If not, un-comment the gdown line.
# RUN uv run gdown 15dEeF3H5uJSPvyod44wLH9ZRKJdrV5Oc
# RUN tar -xvf data.tar.xz -C ../data

# 10. Run the app
# Note: 'uvicorn' is now in the path, so we don't need 'uv run' prefix
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]