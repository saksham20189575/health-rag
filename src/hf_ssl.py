"""TLS setup for HTTPS clients behind the local inspection proxy.

Python 3.13+ turns on ``VERIFY_X509_STRICT``. The Zscaler CA in
``SSL_CERT_FILE`` (``/etc/ssl/certs/ca-bundle.pem``) is a real CA and the
chain verifies, but its basicConstraints extension is not marked critical, so
OpenSSL rejects it. Chain and hostname checks stay enabled. This only drops
that extra flag for clients which explicitly use this context.
"""

from __future__ import annotations

import ssl

_configured = False


def verified_ssl_context() -> ssl.SSLContext:
    """Default trust store, with the strict basicConstraints check cleared."""
    context = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


def hf_ssl_context() -> ssl.SSLContext:
    """Backward-compatible name used by the Hugging Face setup and tests."""
    return verified_ssl_context()


def configure_huggingface_ssl() -> None:
    """Point ``huggingface_hub`` at :func:`hf_ssl_context`. Safe to call twice."""
    global _configured
    if _configured:
        return

    import httpx
    from huggingface_hub.utils import _http

    def client_factory() -> httpx.Client:
        return httpx.Client(
            verify=hf_ssl_context(),
            event_hooks={"request": [_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
        )

    _http.set_client_factory(client_factory)
    _configured = True
