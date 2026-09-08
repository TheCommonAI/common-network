"""Validate and pin outbound destinations before opening a socket.

No proxy environment, redirects, or second hostname resolution. TLS still
verifies the original hostname through httpcore's sni_hostname extension.
"""
import asyncio
import ipaddress
import socket
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
from fastapi import HTTPException
from app.config import settings


def endpoint_identity(url: str) -> str:
    try:
        p = urlsplit(url)
        if (p.scheme not in {'http', 'https'} or not p.hostname or p.username
                or p.password or p.query or p.fragment or '\\' in url
                or any(ord(c) < 33 or ord(c) == 127 for c in url)):
            raise ValueError()
        path = unquote(p.path)
        if '%' in path or '\\' in path or any(x in {'.', '..'} for x in path.split('/')):
            raise ValueError()
        port = p.port or (443 if p.scheme == 'https' else 80)
        host = p.hostname.encode('idna').decode().lower()
        host = f'[{host}]' if ':' in host else host
        return urlunsplit((p.scheme, f'{host}:{port}', p.path.rstrip('/'), '', ''))
    except (ValueError, UnicodeError):
        raise HTTPException(400, 'endpoint_url must be an absolute HTTP(S) URL without credentials, query, fragment or ambiguous path') from None


def _resolved_addresses(host: str):
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        try:
            return list(dict.fromkeys(ipaddress.ip_address(i[4][0])
                         for i in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)))
        except (socket.gaierror, ValueError):
            raise HTTPException(400, 'endpoint host could not be resolved') from None


def validate_addresses(url: str):
    canonical = endpoint_identity(url)
    p = urlsplit(canonical)
    addresses = _resolved_addresses(p.hostname)
    if not addresses:
        raise HTTPException(400, 'endpoint host has no addresses')
    cidrs = [ipaddress.ip_network(c.strip()) for c in settings.allowed_node_cidrs.split(',') if c.strip()]
    for original in addresses:
        a = original.ipv4_mapped if isinstance(original, ipaddress.IPv6Address) and original.ipv4_mapped else original
        if a.is_link_local or a.is_multicast or a.is_unspecified or (a.is_reserved and not a.is_loopback):
            raise HTTPException(400, 'endpoint address is forbidden')
        private_exception = ((a.is_loopback and settings.allow_loopback_node_endpoints)
                             or any(a in net for net in cidrs))
        if not a.is_global and not private_exception:
            raise HTTPException(400, 'private endpoints require an explicit ALLOWED_NODE_CIDRS or loopback opt-in')
        if p.scheme != 'https' and not private_exception:
            raise HTTPException(400, 'public node endpoints require HTTPS')
    return addresses


def validate_endpoint_url(url: str) -> None:
    validate_addresses(url)


async def pinned_request(client, method: str, url: str, **kwargs):
    try:
        addresses = await asyncio.wait_for(asyncio.to_thread(validate_addresses, url), 5)
    except (HTTPException, asyncio.TimeoutError) as exc:
        raise httpx.ConnectError('node destination rejected or DNS timed out') from exc
    original = httpx.URL(url)
    pinned = original.copy_with(host=str(addresses[0]))
    headers = dict(kwargs.pop('headers', {}))
    headers['Host'] = original.netloc.decode('ascii')
    # Numeric-IP pooling must not reuse a TLS connection for a different
    # original hostname sharing that IP. Use a fresh HTTP/1.1 connection.
    headers['Connection'] = 'close'
    return client.build_request(method, pinned, headers=headers,
                                extensions={'sni_hostname': original.host}, **kwargs)
