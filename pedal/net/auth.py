"""Shared-secret bearer token for the block server.

The token comes from the environment variable PEDAL_BLOCK_TOKEN, or from a file
named with --token-file. Never both. The server and the client read it the same
way. Every request carries "Authorization: Bearer <token>", and the server compares
it with hmac.compare_digest. A missing or wrong token gets 401.

The token is never printed or put in an error message.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

TOKEN_ENV = "PEDAL_BLOCK_TOKEN"
MIN_TOKEN_CHARS = 16


class TokenError(ValueError):
    """The token source is unusable. The message never includes the token."""


def check_token(token: str) -> str:
    if not isinstance(token, str) or len(token) < MIN_TOKEN_CHARS or any(c.isspace() for c in token):
        raise TokenError(f"the block token must be at least {MIN_TOKEN_CHARS} characters with no spaces")
    return token


def load_token(token_file: str | Path | None = None, environ=None) -> str | None:
    """The configured token, or None when neither source is set."""
    env = os.environ if environ is None else environ
    from_env = env.get(TOKEN_ENV)
    if token_file is not None:
        if from_env is not None and from_env.strip():
            raise TokenError(f"set either {TOKEN_ENV} or --token-file, not both")
        try:
            text = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError:
            raise TokenError("the token file cannot be read") from None
    else:
        text = (from_env or "").strip()
    if not text:
        return None
    return check_token(text)


def authorization_headers(token: str | None) -> dict:
    return {} if token is None else {"Authorization": f"Bearer {token}"}


def token_matches(token: str, header: str | None) -> bool:
    if header is None:
        return False
    expected = f"Bearer {token}".encode("utf-8")
    presented = header.encode("latin-1", errors="replace")
    return hmac.compare_digest(expected, presented)
