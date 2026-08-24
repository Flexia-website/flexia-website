#!/bin/bash
set -e

echo "🚀 Building Face Search Pro..."

apt-get update
apt-get install -y cmake build-essential libopenblas-dev liblapack-dev

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p uploads static/landmarks exports data

echo "✅ Build complete!"
