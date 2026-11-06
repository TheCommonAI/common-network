import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { isAuthorized } from "../netlify/edge-functions/gate.ts";

const basic = (user: string, pass: string) => `Basic ${Buffer.from(`${user}:${pass}`).toString("base64")}`;

describe("isAuthorized", () => {
  it("accepts the right password with any username", () => {
    assert.equal(isAuthorized(basic("common", "s3cret"), "s3cret"), true);
    assert.equal(isAuthorized(basic("", "s3cret"), "s3cret"), true);
  });

  it("rejects a wrong or missing password", () => {
    assert.equal(isAuthorized(basic("common", "wrong"), "s3cret"), false);
    assert.equal(isAuthorized(basic("common", "s3cre"), "s3cret"), false);
    assert.equal(isAuthorized(basic("common", "s3cretx"), "s3cret"), false);
    assert.equal(isAuthorized(null, "s3cret"), false);
    assert.equal(isAuthorized("", "s3cret"), false);
  });

  it("rejects other schemes and malformed credentials", () => {
    assert.equal(isAuthorized("Bearer s3cret", "s3cret"), false);
    assert.equal(isAuthorized("Basic !!!not-base64", "s3cret"), false);
    assert.equal(isAuthorized(`Basic ${Buffer.from("nocolon").toString("base64")}`, "s3cret"), false);
  });

  it("handles passwords with colons and non-ASCII characters", () => {
    assert.equal(isAuthorized(basic("u", "a:b:c"), "a:b:c"), true);
    assert.equal(isAuthorized(basic("u", "café-ü"), "café-ü"), true);
  });
});
