"""docs/ltx-binding-notes.md and the VERIFY-ON-INSTANCE markers in the code match one to one."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTES = (ROOT / "docs" / "ltx-binding-notes.md").read_text(encoding="utf-8")
PACKAGE = ROOT / "model_server" / "ltx_server"
MARKER = re.compile(r"#\s*VERIFY-ON-INSTANCE:\s*(\d+)")


def markers() -> dict[int, list[str]]:
    found: dict[int, list[str]] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = MARKER.search(line)
            if match:
                found.setdefault(int(match[1]), []).append(f"{path.name}:{line_number}")
    return found


def items() -> dict[int, str]:
    parts = re.split(r"^### (\d+)\. ", NOTES, flags=re.MULTILINE)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def test_every_marker_has_a_checklist_item_and_every_item_has_a_marker() -> None:
    assert set(markers()) == set(items()), (sorted(markers()), sorted(items()))


def test_items_are_numbered_without_gaps() -> None:
    numbers = sorted(items())
    assert numbers == list(range(1, len(numbers) + 1))


def test_markers_only_appear_in_the_ltx_facing_modules() -> None:
    where = {location.split(":")[0] for places in markers().values() for location in places}
    assert where <= {"ltx_binding.py", "ltx_renderer.py", "resident_binding.py"}


def test_every_item_names_its_code_the_files_to_open_and_what_to_confirm() -> None:
    for number, text in items().items():
        for label in (
            "**Assumption:**",
            "**Code:**",
            "**Open:**",
            "**Confirm:**",
            "**If different:**",
        ):
            assert label in text, f"item {number} lacks {label}"
        assert "packages/" in text or "render_test.sh" in text or "torch" in text, number


def test_the_notes_say_how_to_record_the_instances_commit_and_which_commit_was_read() -> None:
    assert "git -C ~/LTX-2 log -1" in NOTES
    assert re.search(r"commit\s+`[0-9a-f]{40}`", NOTES)
    assert "never been run against a real" in NOTES


def test_the_markers_in_the_code_are_not_orphans_of_removed_assumptions() -> None:
    for number, places in markers().items():
        assert places, number
    assert max(markers()) == len(items())
