import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { handle, parseLimit, parseNewMessage } from "../netlify/functions/messages.mts";

type Row = { id: number; name: string; body: string; created_at: string };

function fakeDb() {
  const rows: Row[] = [];
  const queries: { text: string; values: unknown[] }[] = [];
  return {
    rows,
    queries,
    sql<T>(strings: TemplateStringsArray, ...values: unknown[]): Promise<T[]> {
      const text = strings.join("?");
      queries.push({ text, values });
      if (/^\s*INSERT/i.test(text)) {
        const [name, body] = values as [string, string];
        const row: Row = { id: rows.length + 1, name, body, created_at: "2026-10-29T00:00:00.000Z" };
        rows.push(row);
        return Promise.resolve([row] as T[]);
      }
      const limit = values[0] as number;
      const newestFirst = [...rows].reverse().slice(0, limit);
      return Promise.resolve(newestFirst as T[]);
    },
  };
}

function post(body: unknown, headers: Record<string, string> = {}): Request {
  const raw = typeof body === "string" ? body : JSON.stringify(body);
  return new Request("https://board.test/api/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: raw,
  });
}

function get(query = ""): Request {
  return new Request(`https://board.test/api/messages${query}`);
}

describe("parseNewMessage", () => {
  it("defaults a blank name to anon and trims the body", () => {
    assert.deepEqual(parseNewMessage({ name: "   ", body: "  hi  " }), { name: "anon", body: "hi" });
    assert.deepEqual(parseNewMessage({ body: "hi" }), { name: "anon", body: "hi" });
  });

  it("requires a body", () => {
    assert.throws(() => parseNewMessage({ name: "a", body: "   " }), /write something first/);
    assert.throws(() => parseNewMessage({ name: "a" }), /write something first/);
  });

  it("enforces length limits", () => {
    assert.throws(() => parseNewMessage({ name: "a".repeat(33), body: "ok" }), /under 32/);
    assert.throws(() => parseNewMessage({ body: "b".repeat(501) }), /under 500/);
    assert.equal(parseNewMessage({ name: "a".repeat(32), body: "b".repeat(500) }).body.length, 500);
  });

  it("rejects non-text fields, arrays and NUL bytes", () => {
    assert.throws(() => parseNewMessage({ name: ["a"], body: "ok" }), /must be text/);
    assert.throws(() => parseNewMessage(["body"]), /send an object/);
    assert.throws(() => parseNewMessage(null), /send an object/);
    assert.throws(() => parseNewMessage({ body: "bad\u0000byte" }), /not allowed/);
  });
});

describe("parseLimit", () => {
  it("defaults, clamps and validates", () => {
    assert.equal(parseLimit(null), 100);
    assert.equal(parseLimit("5"), 5);
    assert.equal(parseLimit("0"), 1);
    assert.equal(parseLimit("-3"), 1);
    assert.equal(parseLimit("100000"), 200);
    assert.throws(() => parseLimit("abc"), /must be a number/);
    assert.throws(() => parseLimit(""), /must be a number/);
  });
});

describe("handle", () => {
  it("lists nothing on an empty board", async () => {
    const res = await handle(get(), fakeDb());
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { messages: [] });
  });

  it("stores a message and returns it with 201", async () => {
    const db = fakeDb();
    const res = await handle(post({ name: "ana", body: "meet at the gate" }), db);
    assert.equal(res.status, 201);
    const { message } = await res.json();
    assert.equal(message.id, 1);
    assert.equal(message.name, "ana");
    assert.equal(message.body, "meet at the gate");
    assert.equal(db.rows.length, 1);
  });

  it("lists messages newest first and passes the limit to the query", async () => {
    const db = fakeDb();
    await handle(post({ name: "ana", body: "first" }), db);
    await handle(post({ name: "ben", body: "second" }), db);
    const res = await handle(get("?limit=1"), db);
    const { messages } = await res.json();
    assert.deepEqual(messages.map((m: Row) => m.body), ["second"]);
    assert.equal(db.queries.at(-1)?.values[0], 1);
  });

  it("keeps message text as plain data", async () => {
    const db = fakeDb();
    const res = await handle(post({ body: "<img src=x onerror=alert(1)>" }), db);
    assert.equal((await res.json()).message.body, "<img src=x onerror=alert(1)>");
  });

  it("returns 400 for bad input", async () => {
    const db = fakeDb();
    const blank = await handle(post({ name: "a", body: "  " }), db);
    assert.equal(blank.status, 400);
    assert.match((await blank.json()).error, /write something first/);

    assert.equal((await handle(post("not json"), db)).status, 400);
    assert.equal((await handle(post([1, 2]), db)).status, 400);
    assert.equal((await handle(get("?limit=abc"), db)).status, 400);
    assert.equal(db.rows.length, 0);
  });

  it("returns 413 for oversized payloads", async () => {
    const res = await handle(post({ body: "x".repeat(5000) }), fakeDb());
    assert.equal(res.status, 413);
  });

  it("rejects other methods", async () => {
    const req = new Request("https://board.test/api/messages", { method: "DELETE" });
    assert.equal((await handle(req, fakeDb())).status, 405);
  });

  it("marks responses as not cacheable", async () => {
    const res = await handle(get(), fakeDb());
    assert.equal(res.headers.get("Cache-Control"), "no-store");
  });
});
