"""Bounded process-local buckets. Authenticated requests use verified credentials.

Run one gateway process for these quotas; multiple processes need shared state.
Forwarded addresses are accepted only from explicitly trusted proxy CIDRs.
"""
import hashlib
import ipaddress
import secrets
import time
from app.config import settings

_buckets: dict[str, tuple[float, float]] = {}
_salt = secrets.token_bytes(32)


def fingerprint(value):
    return hashlib.blake2b(value.encode(), key=_salt, digest_size=16).hexdigest()


def client_key(headers, client_host):
    peer = client_host or 'unknown'
    networks = [ipaddress.ip_network(c.strip()) for c in settings.trusted_proxy_cidrs.split(',') if c.strip()]
    def trusted(value):
        try:
            return any(ipaddress.ip_address(value) in n for n in networks)
        except ValueError:
            return False
    if not trusted(peer):
        return peer
    chain = [s.strip() for s in headers.get('x-forwarded-for', '').split(',') if s.strip()]
    for value in reversed(chain):
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return peer
        peer = value
        if not trusted(peer):
            break
    return peer


def _check(key, now, rate=None):
    rate = settings.rate_limit_requests_per_minute if rate is None else rate
    if rate <= 0:
        return 0.0
    # Expire inactive buckets and bound memory even under many forged identities.
    if len(_buckets) >= 4096:
        for k, (_, last) in list(_buckets.items()):
            if now - last > 120:
                _buckets.pop(k, None)
        if key not in _buckets and len(_buckets) >= 4096:
            return 60.0
    capacity = float(rate)
    tokens, last = _buckets.get(key, (capacity, now))
    tokens = min(capacity, tokens + max(0.0, now-last) * rate / 60)
    if tokens < 1 - 1e-9:
        _buckets[key] = (tokens, now)
        return (1-tokens) * 60 / rate
    _buckets[key] = (max(0.0, tokens-1), now)
    return 0.0


def check_rate_limit(request, now=None):
    verified = getattr(getattr(request, 'state', None), 'verified_token', None)
    key = 'member:' + fingerprint(verified) if verified else 'ip:' + fingerprint(
        client_key(request.headers, request.client.host if request.client else None))
    return _check(key, time.monotonic() if now is None else now)


def check_action(request, action, rate):
    from fastapi import HTTPException
    key = action + ':' + fingerprint(client_key(request.headers, request.client.host if request.client else None))
    wait = _check(key, time.monotonic(), rate)
    if wait:
        raise HTTPException(429, 'too many requests; retry later', headers={'Retry-After': str(int(wait)+1)})
