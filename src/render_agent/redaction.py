from __future__ import annotations

import re

_SIGV4_AUTHORIZATION = re.compile(
    r"AWS4-HMAC-SHA256\s+Credential=[^\s,]+,\s*SignedHeaders=[^\s,]+,\s*Signature=[0-9a-fA-F]+"
)
_SENSITIVE_KEY_VALUE = re.compile(
    r"(?P<key>\b(?:authorization|x-amz-security-token|aws_session_token|aws_secret_access_key"
    r"|aws_access_key_id)\b[\"']?\s*[:=]\s*)(?P<value>(?:(?:bearer|basic)\s+)?(?:\"[^\"]*\"|'[^']*'|\S+))",
    re.IGNORECASE,
)
_FUNCTION_URL_HOST = re.compile(r"[A-Za-z0-9-]+\.lambda-url\.[a-z0-9-]+\.on\.aws", re.IGNORECASE)
_ACCESS_KEY_ID = re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA|AIPA)[A-Z0-9]{16}\b")
# Session tokens are several hundred base64 characters; nothing else we log is that long.
_TOKEN_LIKE = re.compile(r"[A-Za-z0-9/+=_-]{100,}")


def redact(text: str) -> str:
    text = _SIGV4_AUTHORIZATION.sub("<redacted-authorization>", text)
    text = _SENSITIVE_KEY_VALUE.sub(lambda m: f"{m['key']}<redacted>", text)
    text = _FUNCTION_URL_HOST.sub("<redacted-host>", text)
    text = _ACCESS_KEY_ID.sub("<redacted-key-id>", text)
    return _TOKEN_LIKE.sub("<redacted-token>", text)
