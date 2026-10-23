FROM python:3.11-slim

WORKDIR /app

COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/app ./app
# The default network's manifest and seed list, served at /manifest and /seeds.
COPY commonnet/networks ./networks

EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
