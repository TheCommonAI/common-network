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

Anyone who can reach the page can post, and there is no sign-in.

## Tests

```bash
cd board && npm test
```
