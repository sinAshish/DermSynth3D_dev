# 1. Use Python 3.10 (Required for Pytorch3D compatibility)
FROM python:3.10-slim

# 2. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set up a virtual environment in /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv /opt/venv

# 4. Use /app as the working directory (Avoids confusion with 'webapp' folder)
WORKDIR /app

# 5. Install system compilers (Git + GCC + G++)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 6. STEP 1: Install Build Tools (Standard PyPI)
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

# 10. CRITICAL: Copy the ENTIRE repository root to /app
# This ensures 'dermsynth3d', 'skin3d', and 'webapp' are all present.
COPY . /app

# 11. Fix Import Paths
# Add /app (project root) and /app/skin3d to PYTHONPATH
# This fixes "ModuleNotFoundError: No module named 'dermsynth3d'"
ENV PYTHONPATH="/app:/app/skin3d:$PYTHONPATH"

# 12. Run the App
# We point to the location of main.py relative to the WORKDIR (/app).
# Based on your logs, main.py seems to be in webapp/api/main.py or api/main.py.
# If your main.py is in 'webapp/api/main.py', use this:
CMD ["uvicorn", "webapp.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# IF your main.py is actually in the root (as per your tree structure), use this instead:
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]