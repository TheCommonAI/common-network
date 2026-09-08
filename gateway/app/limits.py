"""Bound request memory, connection lifetime and concurrency, including streams."""
import asyncio
from fastapi import HTTPException
from starlette.responses import JSONResponse
from app.config import settings


def validate_chat(body):
    if not isinstance(body, dict):
        raise HTTPException(400, 'chat body must be a JSON object')
    import math
    for field in ('temperature', 'top_p', 'frequency_penalty', 'presence_penalty'):
        if field in body and (type(body[field]) not in (int, float) or not math.isfinite(body[field])):
            raise HTTPException(400, 'sampling options must be finite numbers')
    messages = body.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= settings.max_messages:
        raise HTTPException(400, 'messages must be a non-empty bounded list')
    for m in messages:
        if (not isinstance(m, dict) or m.get('role') not in {'system', 'developer', 'user', 'assistant', 'tool'}
                or not isinstance(m.get('content'), str)):
            raise HTTPException(400, 'messages must have a supported role and text content')
    if 'stream' in body and type(body['stream']) is not bool:
        raise HTTPException(400, 'stream must be a boolean')
    for field in ('max_tokens', 'max_completion_tokens'):
        value = body.get(field, settings.max_output_tokens)
        if type(value) is not int or not 1 <= value <= settings.max_output_tokens:
            raise HTTPException(400, f'{field} must be within the output token limit')
    if 'max_tokens' not in body and 'max_completion_tokens' not in body:
        body['max_tokens'] = settings.max_output_tokens
    # The network supports text inference. Arbitrary vendor options, URL input,
    # tools and user identifiers are not silently forwarded to strangers.
    allowed = {'model', 'messages', 'stream', 'stream_options', 'temperature', 'top_p',
               'seed', 'stop', 'max_tokens', 'max_completion_tokens', 'frequency_penalty',
               'presence_penalty', 'response_format', 'logprobs', 'top_logprobs'}
    if set(body) - allowed:
        raise HTTPException(400, 'unsupported chat options')
    return body


class RequestLimits:
    def __init__(self, app):
        self.app = app
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        if self.active >= settings.max_active_requests:
            return await JSONResponse({'detail': 'gateway busy; retry later'}, 503)(scope, receive, send)
        self.active += 1
        started = False
        async def safe_send(message):
            nonlocal started
            if message['type'] == 'http.response.start':
                started = True
                headers = list(message.get('headers', []))
                headers.extend([(b'cache-control', b'no-store'), (b'referrer-policy', b'no-referrer'),
                                (b'x-content-type-options', b'nosniff'),
                                (b'content-security-policy', b"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")])
                message['headers'] = headers
            await send(message)
        try:
            async with asyncio.timeout(settings.request_deadline_seconds):
                headers = dict(scope.get('headers', []))
                if headers.get(b'content-encoding', b'identity') != b'identity':
                    return await JSONResponse({'detail': 'compressed request bodies are unsupported'}, 415)(scope, receive, safe_send)
                try:
                    declared = int(headers.get(b'content-length', b'0'))
                except ValueError:
                    declared = -1
                if declared < 0 or declared > settings.max_request_bytes:
                    return await JSONResponse({'detail': 'request body too large or invalid length'}, 413)(scope, receive, safe_send)
                # Bound bytes before FastAPI/Pydantic parse JSON (including chunked requests).
                body = bytearray()
                async with asyncio.timeout(15):
                    while True:
                        message = await receive()
                        if message['type'] == 'http.disconnect':
                            return
                        body.extend(message.get('body', b''))
                        if len(body) > settings.max_request_bytes:
                            return await JSONResponse({'detail': 'request body too large'}, 413)(scope, receive, safe_send)
                        if not message.get('more_body', False):
                            break
                supplied = False
                async def replay():
                    nonlocal supplied
                    if not supplied:
                        supplied = True
                        return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                    return await receive()
                await self.app(scope, replay, safe_send)
        except TimeoutError:
            if not started:
                await JSONResponse({'detail': 'request deadline exceeded'}, 504)(scope, receive, safe_send)
            else:
                # An incomplete stream must not receive a synthetic success terminator.
                raise
        finally:
            self.active -= 1
