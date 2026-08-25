FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Use pip-installed CMake (older, compatible version) instead of the system
# apt package - newer system CMake versions reject dlib's old
# cmake_minimum_required() declaration.
RUN pip install --upgrade pip && \
    pip install "cmake<3.31"

# Extra safety net: allow old CMake policies even if a newer CMake is used.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# dlib compiles from source here - this step can take several minutes
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads static/landmarks exports data

ENV PYTHONUNBUFFERED=1
ENV PORT=5000

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "2", "app:app"]
