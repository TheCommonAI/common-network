# Gateway

A small signpost service, deployed on Railway from `gateway/Dockerfile` (see `railway.json`). It does not run
models and does not route requests. Its job is to help `common join` find peers without a typed address.

It serves:

- `GET /health`: is the service up.
- `GET /manifest` and `GET /seeds`: the default network's manifest and seed list, read from `commonnet/networks/`.
- `POST /announce`, `GET /directory`, `POST /check`: the rendezvous directory. Readers verify every record's signature,
  so the directory does not have to be trusted.
- `GET /api/network-stats`: public, browser-readable nodes, models, layer coverage, and rolling chat timings.
- `POST /api/network-report`: signed, text-free node heartbeats and timing buckets from updated peers.

See [the website API contract and JavaScript example](../docs/network-dashboard.md). No additional database
or API key is needed. Directory and dashboard data are bounded in memory; use a single gateway worker.

Everything else returns `410 Gone`. The old whole-model gateway (v0.1) has been retired; see
[../docs/retiring-the-old-network.md](../docs/retiring-the-old-network.md).

## Running the tests

```bash
cd gateway
python -m unittest discover tests
```

The tests import FastAPI, so install `requirements.txt` first.
