# Mobile worker protocol

This is a network-side contract for Android and iOS runtimes; it does not
implement model download or inference in either mobile app. A compliant app
runs inference locally and initiates an authenticated WebSocket to the gateway.
The device opens no inbound port, publishes no endpoint, and does not need a
tunnel.

## Catalogue and registration

`POST /assign` accepts the existing `hardware` profile plus `platform` set to
`android` or `ios` and runtime capabilities, for example:

```json
{
  "platform": "ios",
  "runtime_capabilities": ["gguf:q2_0", "ternary:hadamard-activations"]
}
```

The result is restricted to entries explicitly marked `android_supported` or
`ios_supported` for the requested platform in `catalogue/catalogue.seed.yaml`,
whose `runtime_requirements` are all present in the supplied capabilities and
whose existing RAM/GPU requirements fit. It returns `weight_format`,
requirements, and `runtime_notes`. Capability values are runtime claims, not
hardware attestation; eligibility does not attest a working device runtime.

The current supported entries and declared formats are:

| Catalogue entry | Weight format | Required capabilities |
|---|---|---|
| `bonsai2-2b`, `bonsai2-4b`, `bonsai2-9b` | `Q2_0` | `gguf:q2_0`, `ternary:hadamard-activations` |
| `bonsai-8b` (first generation) | `Q2_0_g64` | `gguf:q2_0_g64` |

The source URL and checksum in the live catalogue remain authoritative; a
reported model name or filename cannot replace them. The Bonsai 2 `Q2_0`
packing requires the Hadamard ternary runtime path; stock upstream llama.cpp
needs build b11555 or newer. Its `PQ2_0` and `PTQ1_0` packings are not supported
by stock upstream llama.cpp. Bonsai 8B must use the `_g64` source: the ordinary
`Q2_0` packing is fork-only. A runtime that cannot meet the catalogue
requirements must not advertise that capability or register that model.

Register with the ordinary `POST /nodes` endpoint, setting `transport` to
`android_ws` or `ios_ws` to match the platform, omitting `endpoint_url`, and
setting both `model_name` and `catalogue_id` to the assigned catalogue ID.
Include `runtime_name`, `runtime_version`, `model_format`, and the same
`runtime_capabilities` used for
assignment. The gateway rejects unlisted models/formats, mismatched catalogue
IDs, missing runtime requirements, API-key refs, and HTTP worker tokens.
Registration returns the usual node-scoped `node_token`. Existing-name
re-registration requires that token and safely resets health until a new
channel completes its handshake.

## Channel authentication and lease

Connect to `wss://<gateway>/nodes/<node-id>/connect`, passing the token only as
`Authorization: Bearer <node_token>` (or `X-Common-Node-Token`), never in the
URL. The gateway verifies the node ID, mobile transport, and token before accepting
the socket. Android and iOS use the same protocol-v1 `hello`, heartbeat, and
inference frames. The first frame must be a JSON `hello` matching the
registered model, format, runtime name/version, and capabilities:

```json
{
  "type": "hello",
  "protocol_version": 1,
  "model_name": "bonsai2-4b",
  "model_format": "Q2_0",
  "runtime_name": "example-runtime",
  "runtime_version": "1.2.3",
  "runtime_capabilities": ["gguf:q2_0", "ternary:hadamard-activations"]
}
```

The gateway answers `ready` with the negotiated protocol version, a 20-second
heartbeat interval, 60-second lease, one-request concurrency limit, and
64-KiB maximum WebSocket message size. Send `{"type":"heartbeat"}` at least
every 20 seconds, including while inference is running. The app must tolerate
socket closure, OS suspension/termination, radio changes, and reconnects; the
gateway drops health after lease expiry and requires a fresh authenticated
connection. This protocol makes no promise that Android or iOS will keep an app running
in the background.

Use `DELETE /nodes/{id}` with `X-Common-Node-Token` for explicit
deregistration. It removes the registry entry and closes its active channel.
The same node token continues to be the existing contributor credential for
`/v1/chat/completions`, so the contribution gate and request rate limiter are
unchanged.

## Inference framing

The gateway pins every request to the registered model; the device is not a
general-purpose proxy. One request may be in flight per device. A second request
is rejected as busy so the existing gateway candidate fallback can choose a
different healthy node rather than queueing unbounded work.

The gateway emits a bounded `request.start` frame (request UUID, protocol
version, pinned model/format, stream flag, and JSON-body byte length), then
ordered `request.data` frames containing base64 chunks of at most 8 KiB, then
`request.end`. Reassemble exactly the declared number of bytes and parse one
JSON object. The total request body is at most 1 MiB. The gateway sets the
registered model, clamps `max_tokens` to 1–1024, removes the alternate
`max_completion_tokens` limit, and forces `n: 1`.

For streaming requests, reply with ordered frames:

```json
{"type":"response.chunk","request_id":"<uuid>","sequence":0,"chunk":{"id":"...","choices":[{"delta":{"content":"hello"}}]}}
{"type":"response.done","request_id":"<uuid>"}
```

Each `chunk` is an OpenAI-compatible chat-completion chunk object; the gateway
relays each as an SSE `data:` event and terminates with `data: [DONE]`. Use
zero-based contiguous sequence numbers. For a non-streaming request, send `response.result.start` with the JSON byte
length, ordered `response.result.data` frames containing base64 chunks of at
most 8 KiB, then `response.result.end`; the assembled object must be valid
OpenAI-compatible JSON. `response.error` carries a short error message. Frames
must repeat the exact request UUID; unknown or late IDs are ignored and cannot
complete a different request.

If the client disconnects, the read timeout elapses, or the stream is
abandoned, the gateway sends `request.cancel` for that UUID and releases the
slot. The device must cancel inference and discard any late output. Response
queues are bounded to eight frames; a full queue applies WebSocket backpressure.
Individual frames are capped at 64 KiB and generated response bytes at 1 MiB.
Malformed frames, invalid sequence numbers, oversized messages, or a runtime
handshake mismatch close the channel rather than being treated as success.

## Deployment boundary

The gateway currently runs as one Uvicorn worker and the WebSocket broker is
process-local. It safely routes requests only when the device connection and
inference request reach that same gateway process. Do not horizontally scale
gateway workers/replicas for mobile nodes without adding a shared broker and
connection ownership/routing; the database health row is not that broker.
HTTP/Cloudflare-tunnel contributors keep their existing endpoint, worker
authentication, health probes, and streaming path.

The current `common-android` app is a chat client: it sends requests to the
gateway and does not download models, register a node, or infer locally. The
`common-ios` repository currently has no runtime implementation beyond its
README. Both apps still need a contribution client and a Bonsai-capable local
inference runtime before mobile contribution is available to users. iOS
eligibility means the backend will accept a matching runtime capability claim;
it does not assert that the current iOS app or any particular iOS runtime has
been built or validated.
