import type { Config, Context } from "@netlify/edge-functions";

const REALM = 'Basic realm="common message board", charset="UTF-8"';

function constantTimeEqual(a: string, b: string): boolean {
  const left = new TextEncoder().encode(a);
  const right = new TextEncoder().encode(b);
  let diff = left.length ^ right.length;
  for (let i = 0; i < Math.max(left.length, right.length); i++) {
    diff |= (left[i] ?? 0) ^ (right[i] ?? 0);
  }
  return diff === 0;
}

export function isAuthorized(header: string | null, expected: string): boolean {
  if (!header?.startsWith("Basic ")) return false;
  let decoded: string;
  try {
    const bytes = Uint8Array.from(atob(header.slice(6)), (c) => c.charCodeAt(0));
    decoded = new TextDecoder().decode(bytes);
  } catch {
    return false;
  }
  const separator = decoded.indexOf(":");
  if (separator < 0) return false;
  return constantTimeEqual(decoded.slice(separator + 1), expected);
}

export default async (request: Request, context: Context) => {
  const expected = Netlify.env.get("BOARD_PASSWORD");
  if (!expected) {
    return new Response("the board is not set up yet", { status: 503 });
  }
  if (isAuthorized(request.headers.get("authorization"), expected)) {
    return context.next();
  }
  return new Response("password required", {
    status: 401,
    headers: { "WWW-Authenticate": REALM, "Cache-Control": "no-store" },
  });
};

export const config: Config = {
  path: "/*",
};
