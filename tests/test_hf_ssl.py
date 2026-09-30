"""The Hugging Face client keeps certificate checks and accepts the Zscaler CA."""

from __future__ import annotations

import ssl

from src.hf_ssl import hf_ssl_context


def test_ssl_context_verifies_the_chain_without_the_strict_ca_flag():
    context = hf_ssl_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        assert not context.verify_flags & ssl.VERIFY_X509_STRICT
