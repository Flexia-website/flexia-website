#!/bin/bash
set -e

echo "🚀 Building Face Search Pro..."

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    git

pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

mkdir -p uploads static/landmarks exports data

echo "✅ Build complete!"
