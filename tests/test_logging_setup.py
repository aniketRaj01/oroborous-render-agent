from __future__ import annotations

import io
import logging

import pytest

from render_agent.logging_setup import KeyValueFormatter, RedactionFilter, configure_logging
from render_agent.redaction import redact

# Built at runtime so that no secret scanner sees a literal key-shaped string in the source.
KEY_ID = "AKIA" + "ABCDEFGHIJKLMNOP"
SESSION_TOKEN = "tok" + "Qz9/+" * 60  # long unbroken base64-like run
SIGV4_AUTH = (
    f"AWS4-HMAC-SHA256 Credential={KEY_ID}/20261018/us-east-1/lambda/aws4_request, "
    "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, " + "Signature=" + "ab12" * 16
)


@pytest.mark.parametrize(
    ("text", "forbidden"),
    [
        ("calling https://abc123xyz.lambda-url.us-east-1.on.aws/v1/claim", "abc123xyz"),
        ("host=abc-123.lambda-url.eu-west-2.on.aws", "abc-123"),
        (f"key {KEY_ID} leaked", KEY_ID),
        (f"ASIA key {'ASIA' + 'ABCDEFGHIJKLMNOP'}", "ABCDEFGHIJKLMNOP"),
        (f"token={SESSION_TOKEN}", SESSION_TOKEN[:60]),
        (f"x-amz-security-token: {SESSION_TOKEN}", SESSION_TOKEN[:60]),
        (f"Authorization: {SIGV4_AUTH}", "Signature="),
        (f"headers={{'authorization': '{SIGV4_AUTH}'}}", "ab12ab12"),
        ("aws_secret_access_key=notarealsecretvalue0123456789abcdefgh", "notarealsecretvalue"),
        ("Authorization: Bearer opaque-value", "opaque-value"),
    ],
)
def test_redact_masks_sensitive_shapes(text: str, forbidden: str) -> None:
    assert forbidden not in redact(text)


def test_redact_leaves_ordinary_text_alone() -> None:
    text = "route=/v1/claim status=200 job_id=412 plan_id=253 agent=test-agent-1 duration_ms=120"
    assert redact(text) == text


def capture_logger(name: str) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    configure_logging(logging.INFO, stream)
    return logging.getLogger(name), stream


def test_one_line_key_value_records_with_utc_timestamp() -> None:
    logger, stream = capture_logger("render_agent.test")
    logger.info("job_api_call", extra={"route": "/v1/claim", "status": 200, "attempt": 1})
    line = stream.getvalue()
    assert line.count("\n") == 1
    assert (
        line.startswith("ts=") and " level=INFO logger=render_agent.test msg=job_api_call " in line
    )
    timestamp = line.split()[0].removeprefix("ts=")
    assert timestamp.endswith("Z")
    assert "attempt=1 route=/v1/claim status=200" in line


def test_values_with_spaces_are_quoted_and_newlines_escaped() -> None:
    logger, stream = capture_logger("render_agent.test")
    logger.info("hello world", extra={"detail": 'two\nlines "quoted"'})
    line = stream.getvalue()
    assert line.count("\n") == 1
    assert 'msg="hello world"' in line
    assert 'detail="two\\nlines \\"quoted\\""' in line


def test_filter_redacts_message_args_extras_and_exceptions() -> None:
    logger, stream = capture_logger("render_agent.test")
    host = "leakyid.lambda-url.us-east-1.on.aws"
    try:
        raise RuntimeError(f"cannot reach {host}")
    except RuntimeError:
        logger.error("failed %s", host, extra={"target": f"https://{host}", "n": 1}, exc_info=True)
    output = stream.getvalue()
    assert "leakyid" not in output
    assert "<redacted-host>" in output
    assert "n=1" in output


def test_filter_masks_key_ids_in_args() -> None:
    logger, stream = capture_logger("render_agent.test")
    logger.warning("creds %s", KEY_ID)
    assert KEY_ID not in stream.getvalue()


def test_filter_and_formatter_are_usable_standalone() -> None:
    record = logging.LogRecord("x", logging.INFO, "f", 1, "see %s", (KEY_ID,), None)
    assert RedactionFilter().filter(record)
    assert KEY_ID not in KeyValueFormatter().format(record)


def test_http_libraries_do_not_log_request_urls() -> None:
    _, stream = capture_logger("render_agent.test")
    for name in ("httpx", "httpcore"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING
    logging.getLogger("httpx").info(
        "HTTP Request: POST https://someid.lambda-url.us-east-1.on.aws/v1/claim"
    )
    assert stream.getvalue() == ""
