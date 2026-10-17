"""docs/model-server-contract.md may only be appended to: earlier text must still be there."""

from __future__ import annotations

import re
from pathlib import Path

DOC = (Path(__file__).resolve().parents[2] / "docs" / "model-server-contract.md").read_text("utf-8")

ORIGINAL_HEADINGS = [
    "# Model server contract (agent side: implemented; server side: to be built)",
    "## Ground rules",
    "## GET /health",
    "## POST /generate",
    "### Errors",
    "## Timeouts and failure behaviour",
    "## Compatibility",
    "## Try it without a GPU",
]

ORIGINAL_STATEMENTS = [
    "The server is **ready** only when `modelLoaded` is `true`.",
    "`outputPath` must equal the requested path exactly",
    "| 500 | `oom` | GPU or host memory exhausted | `oom` |",
    "| 409 | `busy` | A generation is already running; the running one is not disturbed |",
    "Unknown routes answer\n`404` with the same envelope and code `invalid_request`.",
    "A breaking change to this contract requires new paths",
]


def test_the_original_headings_are_all_present_in_their_original_order() -> None:
    headings = re.findall(r"^#{1,3} .+$", DOC, flags=re.MULTILINE)
    positions = [headings.index(h) for h in ORIGINAL_HEADINGS]
    assert positions == sorted(positions)


def test_the_original_rules_are_still_stated() -> None:
    for statement in ORIGINAL_STATEMENTS:
        assert statement in DOC, statement


def test_the_implementation_notes_come_last() -> None:
    headings = re.findall(r"^## .+$", DOC, flags=re.MULTILINE)
    assert headings[-1] == "## Server implementation notes"
    assert DOC.index("## Server implementation notes") > DOC.index("## Try it without a GPU")


def test_the_notes_cover_loading_warm_up_and_the_oom_restart_policy() -> None:
    notes = DOC.partition("## Server implementation notes")[2]
    for expected in (
        "503",
        "model is still loading",
        "warm-up",
        "MAX_CONSECUTIVE_OOMS",
        "status 75",
    ):
        assert expected in notes, expected
