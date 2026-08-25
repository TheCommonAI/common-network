import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import db, embedder
from app.catalogue import router as catalogue_router, seed_catalogue_from_file
from app.config import settings
from app.decisions import router as decisions_router
from app.demand import router as demand_router
from app.gateway import router as gateway_router
from app.health import health_check_loop
from app.registry import router as registry_router
from app.seed import seed_from_file


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Each step is announced before it runs, not after.
    #
    # This startup once hung on `embedder.load()` -- SentenceTransformer
    # checking huggingface.co for a newer snapshot on a container that could
    # not reach it -- and the only evidence anywhere was uvicorn's "Waiting for
    # application startup." followed by silence. Diagnosing it needed an
    # external check of the database's connection count to prove the hang was
    # before db.connect(). A log line would have said so immediately.
    #
    # Announcing *before* rather than after is the whole point: a step that
    # never completes only shows up in the log if its start was already
    # printed.
    print("startup: loading embedding model...", flush=True)
    embedder.load()
    print("startup: connecting to database...", flush=True)
    await db.connect()
    if settings.seed_on_startup:
        print("startup: seeding nodes...", flush=True)
        await seed_from_file()
    print("startup: seeding catalogue...", flush=True)
    await seed_catalogue_from_file()
    health_task = asyncio.create_task(health_check_loop())
    print("startup: ready", flush=True)
    yield
    health_task.cancel()
    await db.disconnect()


app = FastAPI(
    title="Common Network Gateway",
    description=(
        "Permissionless, transparent routing and composition across contributed AI nodes. "
        "A request that spans domains is answered by a panel of specialists in parallel, "
        "verified deterministically, and synthesised into one reply."
    ),
    version="0.1.1",
    lifespan=lifespan,
)

app.include_router(registry_router)
app.include_router(gateway_router)
app.include_router(decisions_router)
app.include_router(catalogue_router)
app.include_router(demand_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


DASHBOARD_PATH = Path(__file__).parent / "static" / "dashboard.html"


@app.get("/dashboard")
async def dashboard():
    return FileResponse(DASHBOARD_PATH)
