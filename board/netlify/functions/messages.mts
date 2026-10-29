import type { Config, Context } from "@netlify/functions";
import { getDatabase } from "@netlify/database";

const NAME_MAX = 32;
const BODY_MAX = 500;
const PAYLOAD_MAX = 4096;
const LIST_DEFAULT = 100;
const LIST_MAX = 200;
const DEFAULT_NAME = "anon";

export interface Message {
  id: number;
  name: string;
  body: string;
  created_at: string;
}

export interface Db {
  sql<T = Record<string, unknown>>(strings: TemplateStringsArray, ...values: unknown[]): PromiseLike<T[]>;
}

class ValidationError extends Error {}

export function clean(value: unknown, limit: number, required: boolean): string {
  const text = value ?? "";
  if (typeof text !== "string") throw new ValidationError("must be text");
  const trimmed = text.trim();
  if (trimmed.includes("\u0000")) throw new ValidationError("that character is not allowed");
  if (required && !trimmed) throw new ValidationError("write something first");
  if (trimmed.length > limit) throw new ValidationError(`keep it under ${limit} characters`);
  return trimmed;
}

export function parseNewMessage(data: unknown): { name: string; body: string } {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new ValidationError("send an object with name and body");
  }
  const fields = data as Record<string, unknown>;
  const name = clean(fields.name, NAME_MAX, false) || DEFAULT_NAME;
  const body = clean(fields.body, BODY_MAX, true);
  return { name, body };
}

export function parseLimit(raw: string | null): number {
  if (raw === null) return LIST_DEFAULT;
  if (!/^-?\d+$/.test(raw)) throw new ValidationError("limit must be a number");
  return Math.min(Math.max(Number.parseInt(raw, 10), 1), LIST_MAX);
}

function json(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
  });
}

async function listMessages(req: Request, db: Db): Promise<Response> {
  const limit = parseLimit(new URL(req.url).searchParams.get("limit"));
  const rows = await db.sql<Message>`
    SELECT id, name, body, created_at FROM messages ORDER BY id DESC LIMIT ${limit}`;
  return json({ messages: rows });
}

async function postMessage(req: Request, db: Db): Promise<Response> {
  const declared = Number(req.headers.get("content-length") ?? 0);
  const raw = await req.text();
  if (declared > PAYLOAD_MAX || new TextEncoder().encode(raw).byteLength > PAYLOAD_MAX) {
    return json({ error: "message is too large" }, 413);
  }
  let data: unknown;
  try {
    data = JSON.parse(raw || "{}");
  } catch {
    return json({ error: "send json" }, 400);
  }
  const { name, body } = parseNewMessage(data);
  const [message] = await db.sql<Message>`
    INSERT INTO messages (name, body) VALUES (${name}, ${body})
    RETURNING id, name, body, created_at`;
  return json({ message }, 201);
}

export async function handle(req: Request, db: Db): Promise<Response> {
  try {
    if (req.method === "GET") return await listMessages(req, db);
    if (req.method === "POST") return await postMessage(req, db);
    return json({ error: "method not allowed" }, 405);
  } catch (err) {
    if (err instanceof ValidationError) return json({ error: err.message }, 400);
    throw err;
  }
}

let database: Db | undefined;

export default async (req: Request, _context: Context) => {
  database ??= getDatabase();
  return handle(req, database);
};

export const config: Config = {
  path: "/api/messages",
  method: ["GET", "POST"],
};
