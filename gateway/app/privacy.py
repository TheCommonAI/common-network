"""Short retention, no request embeddings by default, content-free failure logging."""
import asyncio
from app import db
from app.config import settings


async def purge_expired():
    async with db.pool().acquire() as conn:
        if not settings.retain_request_embeddings:
            await conn.execute('update decisions set request_embed = null, compose_reason = null '
                               'where request_embed is not null or compose_reason is not null')
        await conn.execute('delete from decisions where created_at < now() - make_interval(days => $1)',
                           max(1, settings.decision_retention_days))


async def retention_loop():
    while True:
        try:
            await purge_expired()
        except Exception:
            print('privacy: retention cleanup failed; check database availability', flush=True)
        await asyncio.sleep(3600)
