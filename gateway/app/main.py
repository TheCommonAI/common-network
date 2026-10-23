"""The Common gateway, retired to a signpost.

Everything this service used to do (node registry, routing, composition, the catalogue, the mobile
transport) now happens inside the layer network, with no central server. What is left:

  GET /health      liveness, for the host
  GET /manifest    the default network's manifest
  GET /seeds       peers that are normally up (same list `common join` fetches from the repository)
  GET /            a short page saying where things went
  anything else    410 Gone with a plain sentence, so an old node or app version learns why it stopped working
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

NETWORKS = Path(os.environ.get("COMMON_NETWORKS_DIR", Path(__file__).resolve().parent.parent / "networks"))
MOVED = ("The Common gateway has been retired. The network now runs peer to peer: install or update the "
         "`common` command and run `common join`. See https://github.com/TheCommonAI/common-network")

app = FastAPI(title="Common gateway (retired)", docs_url=None, redoc_url=None, openapi_url=None)


def _load(name: str) -> dict:
    try:
        return json.loads((NETWORKS / name).read_text())
    except (OSError, ValueError):
        return {}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "common-gateway", "mode": "retired"}


@app.get("/manifest")
def manifest() -> dict:
    return _load("main.json")


@app.get("/seeds")
def seeds() -> dict:
    data = _load("seeds.json")
    extra = [s.strip() for s in os.environ.get("COMMON_SEEDS", "").split(",") if s.strip()]
    return {"network": data.get("network"), "seeds": list(dict.fromkeys([*data.get("seeds", []), *extra]))}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Common</title><body style='font-family:system-ui;max-width:34rem;margin:4rem auto;padding:0 1rem'>"
            "<h1>Common</h1><p>The network now runs peer to peer, with no central gateway.</p>"
            "<p>Install or update the <code>common</code> command, then run <code>common join</code>.</p>"
            "<p><a href='https://github.com/TheCommonAI/common-network'>github.com/TheCommonAI/common-network</a></p>")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
async def gone(path: str, request: Request) -> JSONResponse:
    return JSONResponse({"error": MOVED}, status_code=410)
