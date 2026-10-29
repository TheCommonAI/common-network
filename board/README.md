# Message board

A small message board for the Common community: one page, one SQLite file,
no accounts. Standard library only (Python 3.8+).

```bash
python3 board/board.py                      # http://127.0.0.1:8080
python3 board/board.py --host 0.0.0.0       # reachable from other machines on the network
```

Environment: `BOARD_DB` (default `board/board.db`), `BOARD_HOST` (default
`127.0.0.1`), `PORT` (default `8080`). The same settings are available as
`--db`, `--host` and `--port`.

Anyone who can reach the page can post, and there is no sign-in, so run it
only where that is acceptable.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/` | the page |
| `GET` | `/api/messages?limit=N` | newest first; `N` is clamped to 1-200 (default 100) |
| `POST` | `/api/messages` | JSON `{"name": "...", "body": "..."}`; name is optional (defaults to `anon`), body is required |
| `GET` | `/health` | liveness |

Limits: name up to 32 characters, body up to 500, request body up to 4 KB.
Message text is always shown as plain text, never as HTML.

## Tests

```bash
cd board && python3 -m unittest -v test_board
```
