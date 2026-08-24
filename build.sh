#!/bin/bash
set -e

echo "🚀 Building Face Search Pro..."

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libsm6 \
    libxext6 \
    libxrender-dev

pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt

mkdir -p uploads static/landmarks exports data

echo "✅ Build complete!"
