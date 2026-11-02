# Gateway

A small signpost service, deployed on Railway from `gateway/Dockerfile` (see `railway.json`). It does not run
models and does not route requests. Its job is to help `common join` find peers without a typed address.

It serves:

- `GET /health`: is the service up.
- `GET /manifest` and `GET /seeds`: the default network's manifest and seed list, read from `commonnet/networks/`.
- `POST /announce`, `GET /directory`, `POST /check`: the rendezvous directory. Readers verify every record's signature,
  so the directory does not have to be trusted.

Everything else returns `410 Gone`. The old whole-model gateway (v0.1) has been retired; see
[../docs/retiring-the-old-network.md](../docs/retiring-the-old-network.md).

## Running the tests

```bash
cd gateway
python -m unittest discover tests
```

The tests import FastAPI, so install `requirements.txt` first.
