"""TLS for a directly reachable peer: a self-signed certificate per peer, pinned by fingerprint.

A tunnel gave TLS through a public CA. A peer reached at https://IP:PORT has no hostname and no CA
certificate, so trust comes from the peer's signed record instead:

- make_peer_cert() makes (once, then reuses) a self-signed ECDSA P-256 certificate and returns its
  fingerprint, the sha256 of the DER certificate as lowercase hex.
- bind_fingerprint() signs that fingerprint with the peer's ed25519 identity key. A record carries
  {fp, fp_sig}. A client that has already verified the record's own signature (records.verify_record)
  calls verify_binding() to learn that the holder of the identity key vouches for this certificate.
  (The proposed SIGNED_FIELDS change, which would put fp inside the record's own signature, is in
  DIRECT.md. Until then fp_sig carries the binding by itself.)
- client_ssl_context(fp) refuses any server whose certificate does not hash to fp. No CA, no hostname
  check, no expiry check: the pin is the whole decision. A mismatch raises PinMismatch.
- server_ssl_context(cert, key) is the matching server side (TLS 1.2 or newer).

Stdlib ssl plus `cryptography`, which commonnet already needs for record signatures.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import os
import ssl
from pathlib import Path

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.x509.oid import NameOID

CERT_FILE = "peer-cert.pem"
KEY_FILE = "peer-key.pem"
VALID_DAYS = 3650
RENEW_BEFORE_DAYS = 30
BINDING_CONTEXT = b"commonnet-tls-cert-v1\n"


class PinMismatch(ssl.SSLError):
    """The server's certificate is not the one the record pinned."""


def normalize_fingerprint(fp: str) -> str:
    """Lowercase hex with no separators. Raises ValueError unless it is 64 hex characters."""
    if not isinstance(fp, str):
        raise ValueError("fingerprint must be a string")
    text = fp.replace(":", "").strip().lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError("fingerprint must be 64 hex characters (sha256)")
    return text


def fingerprint_of_der(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def fingerprint_of_pem_file(cert_path) -> str:
    cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    return fingerprint_of_der(cert.public_bytes(serialization.Encoding.DER))


def _bind_bytes(fingerprint: str) -> bytes:
    return BINDING_CONTEXT + normalize_fingerprint(fingerprint).encode("ascii")


def bind_fingerprint(identity, fingerprint: str) -> str:
    """base64 ed25519 signature by the identity key over the fingerprint (domain separated)."""
    sign = getattr(identity, "sign", None) or identity._key.sign
    return base64.b64encode(sign(_bind_bytes(fingerprint))).decode("ascii")


def verify_binding(pub_b64: str, fingerprint: str, sig_b64: str) -> bool:
    """True if the key `pub_b64` signed this fingerprint. False for anything malformed."""
    try:
        pub = base64.b64decode(pub_b64.encode("ascii"), validate=True)
        sig = base64.b64decode(sig_b64.encode("ascii"), validate=True)
        Ed25519PublicKey.from_public_bytes(pub).verify(sig, _bind_bytes(fingerprint))
        return True
    except (InvalidSignature, ValueError, TypeError, AttributeError):
        return False


def _usable(cert_path: Path, key_path: Path, now: datetime.datetime) -> bool:
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), None)
    except (OSError, ValueError):
        return False
    same = (cert.public_key().public_numbers() == key.public_key().public_numbers()
            if isinstance(key, ec.EllipticCurvePrivateKey) else False)
    left = cert.not_valid_after_utc - now
    return same and left > datetime.timedelta(days=RENEW_BEFORE_DAYS)


def make_peer_cert(identity, directory) -> tuple:
    """(cert_path, key_path, fingerprint) for this peer. Reuses what is in `directory` while it is
    valid, so the fingerprint stays stable across restarts (other peers have it pinned). The key
    file is created readable by its owner only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = directory / CERT_FILE, directory / KEY_FILE
    now = datetime.datetime.now(datetime.timezone.utc)
    if not _usable(cert_path, key_path, now):
        key = ec.generate_private_key(ec.SECP256R1())
        name_value = "commonnet-" + str(getattr(identity, "peer", "peer"))[:32]
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name_value)])
        cert = (x509.CertificateBuilder()
                .subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=VALID_DAYS))
                .sign(key, hashes.SHA256()))
        key_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())
        tmp = directory / (KEY_FILE + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key_pem)
        os.replace(tmp, key_path)
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_path), str(key_path), fingerprint_of_pem_file(cert_path)


def server_ssl_context(cert_path, key_path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert_path), str(key_path))
    return ctx


class _PinnedContext(ssl.SSLContext):
    """A client context that checks the server certificate's sha256 after the handshake."""

    pin: str = ""

    def wrap_socket(self, sock, *args, **kwargs):
        wrapped = super().wrap_socket(sock, *args, **kwargs)
        try:
            der = wrapped.getpeercert(binary_form=True)
            if not der or not hmac.compare_digest(fingerprint_of_der(der), self.pin):
                raise PinMismatch("server certificate does not match the pinned fingerprint")
        except BaseException:
            wrapped.close()
            raise
        return wrapped


def client_ssl_context(expected_fingerprint: str) -> ssl.SSLContext:
    """Trust exactly the certificate with this sha256 fingerprint, and nothing else."""
    pin = normalize_fingerprint(expected_fingerprint)
    ctx = _PinnedContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # the pin below is the check; wrap_socket enforces it
    ctx.pin = pin
    return ctx
