#!/usr/bin/env bash
# Regenerate requirements.lock from uv.lock, or verify that the committed file is current.
#
#   scripts/export_requirements.sh          rewrite requirements.lock
#   scripts/export_requirements.sh --check  exit 1 if requirements.lock is not what uv exports
#
# The file has hashes, no dev dependencies and not the project itself: the instance installs it
# with `pip install --require-hashes --only-binary=:all: --no-deps` and runs the agent from the
# release's source tree. --locked makes the export fail when uv.lock is stale versus pyproject.toml.
# Keep the uv version in CI in step with the one used here: the output format is uv's.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${repo_root}/requirements.lock"

export_to() {
    (cd "${repo_root}" && uv export --locked --no-dev --no-emit-project \
        --format requirements-txt --no-header --no-annotate --output-file "$1" >/dev/null)
}

case "${1:-}" in
"")
    export_to "${target}"
    echo "wrote ${target}"
    ;;
--check)
    tmp="$(mktemp)"
    trap 'rm -f "${tmp}"' EXIT
    export_to "${tmp}"
    if ! diff -u "${target}" "${tmp}"; then
        echo "requirements.lock is out of date: run scripts/export_requirements.sh and commit it" >&2
        exit 1
    fi
    echo "requirements.lock is in sync with uv.lock"
    ;;
*)
    echo "usage: $0 [--check]" >&2
    exit 2
    ;;
esac
