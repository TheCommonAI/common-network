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
    embedder.load()
    await db.connect()
    if settings.seed_on_startup:
        await seed_from_file()
    await seed_catalogue_from_file()
    health_task = asyncio.create_task(health_check_loop())
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
