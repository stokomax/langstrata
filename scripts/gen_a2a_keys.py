#!/usr/bin/env python
"""Generate the RSA keypair + JWKS document for the a2a demo's emitter.

Writes into ``.a2a-keys/`` (gitignored):

- ``worker-private.pem``      -> point ``A2A_PRIVATE_KEY_FILE`` at this
- ``worker-public.jwks.json`` -> serve at ``A2A_SUBAGENT_JWKS_URL`` in
  ``verify``/``strict`` modes

The a2a demo default is ``A2A_VERIFY_MODE=dev``, which needs no keys at all —
generate them only when you graduate to ``verify``/``strict``.

Run:
    uv run python scripts/gen_a2a_keys.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from langshark_bites.a2a_completion_notifier.signer import A2ASigner


def main() -> None:
    """Write a fresh RSA keypair + JWKS document for the a2a emitter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=".a2a-keys",
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--kid",
        default="worker-1",
        help="key id published in the JWKS (default: %(default)s)",
    )
    parser.add_argument(
        "--issuer",
        default="https://langstrata-worker",
        help="iss claim of the emitter JWT",
    )
    args = parser.parse_args()

    audience = os.environ.get("A2A_RECEIVER_URL", "http://localhost:8001")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    signer = A2ASigner(private_key_pem=pem, kid=args.kid, issuer=args.issuer, audience=audience)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    private_path = out_dir / "worker-private.pem"
    jwks_path = out_dir / "worker-public.jwks.json"
    private_path.write_text(pem, encoding="utf-8")
    jwks_path.write_text(json.dumps(signer.jwks(), indent=2), encoding="utf-8")

    print("A2A demo keys written:")
    print(f"  private key : {private_path}  (set A2A_PRIVATE_KEY_FILE=...)")
    print(f"  jwks        : {jwks_path}  (serve at A2A_SUBAGENT_JWKS_URL)")


if __name__ == "__main__":
    main()
