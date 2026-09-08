-- Reusable worker tokens must remain recoverable for outbound authentication.
-- Node ownership/chat credentials need only verification hashes. Random tokens
-- have sufficient entropy for SHA-256 (this is not password hashing).
update nodes set node_token = 'sha256:' || encode(digest(node_token, 'sha256'), 'hex')
where node_token is not null and node_token not like 'sha256:%';
