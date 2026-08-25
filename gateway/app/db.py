import asyncpg
from pgvector.asyncpg import register_vector

from app.config import settings

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def connect() -> asyncpg.Pool:
    """Open the pool, with every wait bounded.

    asyncpg puts no timeout on connecting or on a query by default. If the
    socket to Postgres dies quietly -- a blip on the provider's private
    network is enough -- an await on it never returns, never raises, and never
    burns CPU. The process simply stops, forever.

    That is not hypothetical. This gateway hung on exactly that: the container
    sat at 651MB and 0% CPU with its log stopped mid-startup, while Postgres
    reported no backend for it at all, because the server had already dropped
    the connection the client was still waiting on.

    Bounded waits turn a silent hang into a loud crash, and a crash gets
    restarted. A hang does not.
    """
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=1,
        max_size=10,
        init=_init_connection,
        # Cap on any single statement. Everything this gateway runs is a small
        # indexed read or a single-row write, so anything near this is a fault
        # rather than slowness.
        command_timeout=settings.db_command_timeout_seconds,
        timeout=settings.db_connect_timeout_seconds,
    )
    return _pool


async def disconnect() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("database pool not initialised — call connect() first")
    return _pool
