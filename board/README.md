# Message board

A small message board for the Common community: one page, messages kept in
Netlify Database, no accounts. The page is static HTML/CSS/JS in `public/`; the
API is one Netlify Function, `netlify/functions/messages.mts`, at `/api/messages`.

## Run it locally

```bash
cd board
npm install
npx netlify dev            # http://localhost:8888
npx netlify database migrations apply   # once, to create the table locally
```

`netlify dev` does not apply migrations itself. Run the apply command after
adding a migration, then restart the dev server if it was already running.

## Password and `/admin-mes`

Every page and API request needs the board password. The browser asks for it
(HTTP Basic auth; any username works, the password is what counts). The check
is in `netlify/edge-functions/gate.ts` and reads the `BOARD_PASSWORD` secret.
If that secret is not set, the board returns 503 and lets nobody in.

Locally, put `BOARD_PASSWORD=...` in `board/.env` (git-ignored). On Netlify, set
it under Site configuration > Environment variables, marked as secret.

The board is served from `/admin-mes/` (with the trailing slash). The redirects
in `netlify.toml` map that path onto the site root.

`commonai.com.au` is served by the existing Netlify project `commonai1`, not by
this site. To show the board at `commonai.com.au/admin-mes/`, add this proxy rule
to `commonai1` (in its `netlify.toml`, or under Site configuration > Redirects),
above any catch-all rule, and deploy `commonai1`:

```
/admin-mes/*  https://common-message-board.netlify.app/admin-mes/:splat  200
```

The browser only talks to `commonai.com.au`, so this also avoids the
`.netlify.app` address at school. This board site must be deployed first.

## Deploy

Netlify applies the committed migration in `netlify/database/migrations/`
during each deploy, before the new version goes live. Do not run migrations
against the hosted database by hand.

Publish settings are in `netlify.toml`: static files from `public/`, functions
from `netlify/functions/`. The site's base directory must be `board`.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/messages?limit=N` | newest first; `N` is clamped to 1-200 (default 100) |
| `POST` | `/api/messages` | JSON `{"name": "...", "body": "..."}`; name is optional (defaults to `anon`), body is required |

Limits: name up to 32 characters, body up to 500, request body up to 4 KB.
Message text is always shown as plain text, never as HTML.

Anyone with the password can post. There are no individual accounts.

## Tests

```bash
cd board && npm test
```
