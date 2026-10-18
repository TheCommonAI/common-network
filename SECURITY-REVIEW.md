# Security review: prompt privacy

**Date:** 2026-10-18  
**Scope:** Query confidentiality in the Common Network Alpha gateway and node
transport.

## Summary

The network is permissionless: anyone can register a node. A node operator can
read the complete prompt when the gateway routes a request to that node. This
does not give a node access to every query, but it means users should not send
confidential prompts to the shared network unless they trust the nodes that may
receive them. The gateway itself also receives and processes prompt text.

HTTPS protects traffic in transit only on connections configured to use it. It
does not conceal a prompt from the gateway or from a selected node, where the
request is processed.

## Findings

### 1. Prompt text is visible to selected node operators

**Severity:** High  
**Confidence:** 9/10  
**Affected code:** `gateway/app/registry.py:201-230,278-303`;
`gateway/app/gateway.py:567,584-626,667`; `gateway/app/upstream.py:74-97`

Node registration is permissionless. The gateway selects healthy nodes and
forwards the request body, including the prompt, to the selected endpoint.
Consequently, an operator controlling a registered node can read prompts routed
to it. This is conditional on selection, panel participation, or fallback; it
is not passive access to all network traffic. The gateway also processes the
prompt in plaintext.

**Recommendation:** Treat every node operator that could receive a request as
a recipient of its prompt. Do not send confidential material to the shared
network; use trusted nodes or a private, access-controlled deployment for
sensitive requests. TLS cannot protect a prompt from the node that must process
it.

### 2. HTTP node transport exposes prompts to on-path observers

**Severity:** Medium  
**Confidence:** 9/10  
**Affected code:** `gateway/app/registry.py:108-137`;
`gateway/app/upstream.py:59-66,70-97`; `join/join.py:1725-1732`;
`join/worker.py:455-484`

The registry permits both HTTP and HTTPS node endpoints, and forwarding uses
the registered scheme. LAN mode registers an HTTP endpoint, while the worker
listens on all interfaces. An attacker able to observe or intercept that
network traffic can read prompts in transit and capture the worker bearer
token. This requires an on-path position; it does not mean every device on a
LAN can automatically read unicast traffic.

**Recommendation:** Require HTTPS for non-local node endpoints, or use HTTP
only on a trusted network or through a protected tunnel/VPN. Avoid LAN mode on
networks where traffic may be observed.

## Scope and limitations

Bundled clients default to an HTTPS gateway, which protects the client-to-
gateway link from passive network observers when used as configured. This
review does not establish the security of a particular deployment, node
operator, custom gateway URL, or network configuration.
