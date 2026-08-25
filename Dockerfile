FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

# Install everything except face-recognition itself first.
# dlib-bin is a prebuilt wheel (no compilation) that provides the same
# "dlib" module as the real dlib package - this avoids the slow/fragile
# source build of dlib that was failing/hanging on Render's build machine.
RUN pip install --no-cache-dir -r requirements.txt

# face-recognition's own metadata requires the package named "dlib"
# (not "dlib-bin"), so install it with --no-deps to skip that check -
# dlib-bin already provides the "dlib" module it needs at import time.
RUN pip install --no-cache-dir --no-deps face-recognition==1.3.0

COPY . .

RUN mkdir -p uploads static/landmarks exports data

ENV PYTHONUNBUFFERED=1
ENV PORT=5000

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "--workers", "2", "app:app"]
