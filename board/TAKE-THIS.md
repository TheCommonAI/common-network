# Take this board and run it yourself

You do not need to give this agent access to `commonai-website`. Use one of
the three options below.

The board lives in this folder. Download it as a zip from GitHub:

https://github.com/TheCommonAI/common-network/archive/refs/heads/cursor/message-board-com-50-be50.zip

Unzip, then use only the `board/` folder (ignore the rest of the repo).

A board-only zip is also attached to the pull request as
`common-message-board.zip`.

Set a password you choose. Do not reuse the example values in this file.

---

## Option 1 — Python, nothing else (fastest)

Needs Python 3.8+. No Netlify, no npm.

```bash
cd board
python3 portable.py --host 0.0.0.0 --port 8080 --password YOUR_PASSWORD
```

Open `http://YOUR_MACHINE:8080/admin-mes/`. The browser asks for a password
(any username, the password is what you set). Messages are saved in
`board/board.db`.

To put it on the public internet, run that command on a host you control
(Railway, a VPS, a home machine with a tunnel) and point a URL at it.

---

## Option 2 — Drop it into `commonai-website`

This is the path to `https://commonai.com.au/admin-mes/`.

1. Copy the whole `board/` folder into `commonai-website` (keep it named `board`,
   at the repo root).
2. In the **website** `netlify.toml` (the one that already deploys
   commonai.com.au), add these rules **above** any catch-all `/*` rule:

```toml
[[redirects]]
  from = "/admin-mes/api/*"
  to = "/.netlify/functions/messages"
  status = 200

[[redirects]]
  from = "/admin-mes"
  to = "/admin-mes/"
  status = 301

[[redirects]]
  from = "/admin-mes/*"
  to = "/board/public/:splat"
  status = 200
```

   The exact function URL depends on how that site is built. If the site
   already uses Netlify Functions from `netlify/functions/`, it is cleaner
   to copy these files instead of the whole folder:

   - `board/public/*` → `admin-mes/` in the website publish directory
   - `board/netlify/functions/messages.mts` → `netlify/functions/messages.mts`
   - `board/netlify/edge-functions/gate.ts` → `netlify/edge-functions/gate.ts`
   - `board/netlify/database/migrations/` → `netlify/database/migrations/`
   - add `@netlify/database` and `@netlify/functions` to that site's
     `package.json`

   A copy-paste starting point is in `commonai-website-redirect.toml`.

3. In the Netlify UI for **commonai1**, set a secret environment variable:
   `BOARD_PASSWORD` = the password you want.
4. Commit and push `commonai-website`. Netlify deploys it the usual way.
5. Open `https://commonai.com.au/admin-mes/`.

If the website's publish directory already has its own `netlify.toml` redirects
that swallow `/admin-mes`, the new rules must be listed first.

---

## Option 3 — New Netlify site from this folder

On a machine logged into the CommonAI Netlify account:

```bash
cd board
npm install
npx netlify login
npx netlify init          # create or link a site; base directory is this folder
npx netlify env:set BOARD_PASSWORD YOUR_PASSWORD --secret
npx netlify deploy --prod
```

Then add this single proxy rule to `commonai-website` (again, above any
catch-all) so school browsers stay on `commonai.com.au`:

```
/admin-mes/*   https://YOUR-SITE.netlify.app/admin-mes/:splat   200
```

---

## Check it

```bash
cd board
python3 -m unittest test_portable.py
npm test
```
