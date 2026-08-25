FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/app ./app
COPY gateway/migrations ./migrations
COPY gateway/nodes.seed.yaml .
COPY catalogue ./catalogue

# Warm the embedding model into the image so first request isn't a cold download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Then refuse to talk to huggingface.co ever again.
#
# Baking the model in was not enough on its own: SentenceTransformer() still
# contacts the Hub at *runtime* to check whether the cached snapshot is stale,
# and that call is made before the app can serve anything. When it hangs, the
# container prints "Waiting for application startup." and nothing else, forever
# -- no error, no timeout, no clue. That is exactly how this presented: a
# gateway stuck in startup with zero database connections, which is what
# narrowed it to the line before db.connect().
#
# Offline mode makes startup depend only on what is already in the image. It
# also means the gateway starts on a restricted network, which matters
# directly: the school-lab deployment runs its own gateway behind a firewall
# that may not allow huggingface.co at all.
#
# If the model were somehow missing, this now fails immediately and loudly
# instead of hanging, which is the better failure.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
