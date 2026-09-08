"""Random node credentials are verified by digest; worker credentials are outbound secrets."""
import hashlib
import secrets


def token_digest(token):
    return 'sha256:' + hashlib.sha256(token.encode()).hexdigest()


def token_matches(stored, supplied):
    if not stored or not supplied:
        return False
    expected = token_digest(supplied) if stored.startswith('sha256:') else supplied
    return secrets.compare_digest(stored.encode(), expected.encode())
