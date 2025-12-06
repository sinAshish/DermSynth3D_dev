# Use Python 3.10 (Stable for ML)
FROM python:3.10-slim

# 1. Install 'uv' by copying the binary from the official image
# This is the Docker best-practice for installing uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /webapp

# 2. Install system compilers required by Pytorch3D
# We need git to fetch the repo and gcc/g++ to compile the C++ extensions
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 3. Install PyTorch CPU version first using uv
# We use '--system' to install into the container's global Python environment
# We install this separately so it is available when Pytorch3D tries to compile
RUN uv pip install --system --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install Pytorch3D directly from source
# This will now succeed because 'torch' is already installed from the step above
RUN uv pip install --system --no-cache "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 5. Install remaining dependencies
COPY requirements.txt .
# IMPORTANT: Ensure 'torch' and 'pytorch3d' are REMOVED from requirements.txt
# to prevent uv from trying to reinstall or downgrade them.
RUN uv pip install --system --no-cache -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Start the application
# Adjust "api.index:app" if your main FastAPI instance is located elsewhere
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]