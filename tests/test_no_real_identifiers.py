"""The repository is public: product files may contain only placeholders and a short allowlist.

Tests are covered by gitleaks and by their own fixtures (obviously fake values); this checks
everything that ships or documents the deployment.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKIPPED_DIRS = {"tests", ".git", ".venv", "node_modules"}
SKIPPED_FILES = {"uv.lock", "requirements.lock"}

# The only URLs product files may contain. The project's own public repository is the bootstrap's
# default source of releases; the others are hook sources, documentation and obvious fakes.
ALLOWED_URLS = {
    "https://github.com/aniketRaj01/oroborous-render-agent.git",
    "https://github.com/astral-sh/ruff-pre-commit",
    "https://github.com/gitleaks/gitleaks",
    "https://github.com/pre-commit/pre-commit-hooks",
    "https://github.com/shellcheck-py/shellcheck-py",
    "https://docs.astral.sh/uv/",
    "https://github.com/Lightricks/LTX-2",
    "https://example.invalid",
    "https://fake-id.lambda-url.us-east-1.on.aws",
    "http://127.0.0.1:8000",
}

FORBIDDEN = {
    "an AWS account id": re.compile(r"\b[0-9]{12}\b"),
    "an ARN with an account": re.compile(r"arn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:[0-9]{12}"),
    "an access key id": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "an EC2 instance id": re.compile(r"\bi-[0-9a-f]{8,17}\b"),
    # the one fake id used by the demo is fine; any other id is a candidate real Function URL
    "a Function URL host": re.compile(
        r"(?<![a-z0-9-])(?!fake-id\.)[a-z0-9-]+\.lambda-url\.[a-z0-9-]+\.on\.aws"
    ),
    "a private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
URL = re.compile(r"https?://[^\s\"'<>)`,]+")
IPV4 = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
ALLOWED_IPS = {"127.0.0.1", "0.0.0.0"}


def product_files() -> list[Path]:
    try:
        listed = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, timeout=30, check=True
        ).stdout.splitlines()
        candidates = [REPO / name for name in listed]
    except (OSError, subprocess.CalledProcessError):
        candidates = [p for p in REPO.rglob("*") if p.is_file()]
    return [
        path
        for path in candidates
        if path.is_file()
        and path.name not in SKIPPED_FILES
        and not SKIPPED_DIRS & set(path.relative_to(REPO).parts)
        and path.suffix not in {".png", ".ico"}
    ]


def read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def test_the_scan_covers_the_deployment_files() -> None:
    names = {str(p.relative_to(REPO)) for p in product_files()}
    for expected in (
        "deploy/bootstrap.py",
        "deploy/install.sh",
        "deploy/systemd/render-agent.service",
        "docs/instance-runbook.md",
        "scripts/release.py",
        "CLAUDE.md",
    ):
        assert expected in names


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_no_forbidden_identifier_shapes(label: str) -> None:
    hits = []
    for path in product_files():
        text = read(path)
        if text is None:
            continue
        for match in FORBIDDEN[label].finditer(text):
            hits.append(f"{path.relative_to(REPO)}: {match.group(0)}")
    assert hits == [], f"{label} found:\n" + "\n".join(hits)


def test_only_allowlisted_urls_appear() -> None:
    unexpected = []
    for path in product_files():
        text = read(path)
        if text is None:
            continue
        for url in URL.findall(text):
            cleaned = url.rstrip(".;:")
            if cleaned not in ALLOWED_URLS and not any(
                cleaned.startswith(allowed + "/") for allowed in ALLOWED_URLS
            ):
                unexpected.append(f"{path.relative_to(REPO)}: {cleaned}")
    assert unexpected == [], "\n".join(unexpected)


def test_only_loopback_ip_addresses_appear() -> None:
    unexpected = []
    for path in product_files():
        text = read(path)
        if text is None:
            continue
        for address in IPV4.findall(text):
            if address not in ALLOWED_IPS:
                unexpected.append(f"{path.relative_to(REPO)}: {address}")
    assert unexpected == [], "\n".join(unexpected)
