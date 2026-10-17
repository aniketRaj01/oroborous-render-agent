#!/usr/bin/env bash
# Prepare a machine (or machine image) to run the render agent. Run once as root; safe to re-run.
#
#   sudo deploy/install.sh [--service-user USER] [--install-packages] [--enable-deadman]
#   deploy/install.sh --root DIR --service-user USER [--dry-run]     # stage into DIR for testing
#
# It creates the directories and config files, copies the bootstrap and the systemd units, and
# enables render-agent.service so it starts at every boot. It does not install the agent itself:
# the bootstrap fetches the newest released version at boot.
set -euo pipefail

usage() {
    cat <<'USAGE'
usage: install.sh [options]

  --service-user USER   account the service runs as (default: $SUDO_USER, else the first
                        regular account on this machine)
  --ltx-root DIR        also install and enable the model server unit; DIR is the LTX-2
                        checkout whose .venv runs it (absolute path)
  --root DIR            stage everything under DIR instead of /, for testing; no systemctl calls
  --dry-run             print what would be done and change nothing
  --install-packages    apt-get install missing prerequisites (nothing is installed by default)
  --enable-deadman      also enable the opt-in dead-man timer (off by default)
  -h, --help            show this help
USAGE
}

die() {
    echo "install.sh: $*" >&2
    exit 1
}

say() {
    echo "install.sh: $*"
}

root=""
dry_run=0
service_user=""
ltx_root=""
install_packages=0
enable_deadman=0

while [ $# -gt 0 ]; do
    case "$1" in
    --service-user)
        [ $# -ge 2 ] || die "--service-user needs a value"
        service_user="$2"
        shift 2
        ;;
    --ltx-root)
        [ $# -ge 2 ] || die "--ltx-root needs a value"
        ltx_root="${2%/}"
        shift 2
        ;;
    --root)
        [ $# -ge 2 ] || die "--root needs a value"
        root="${2%/}"
        shift 2
        ;;
    --dry-run)
        dry_run=1
        shift
        ;;
    --install-packages)
        install_packages=1
        shift
        ;;
    --enable-deadman)
        enable_deadman=1
        shift
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        die "unknown option: $1"
        ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${ltx_root}" ]; then
    case "${ltx_root}" in
    /* | "") ;;
    *) die "--ltx-root must be an absolute path" ;;
    esac
    case "${ltx_root}" in
    *[!A-Za-z0-9_./-]*) die "--ltx-root may only contain letters, digits and _ . / -" ;;
    esac
fi

if [ -n "${root}" ]; then
    [ "${install_packages}" -eq 0 ] || die "--install-packages cannot be combined with --root"
    if [ "${dry_run}" -eq 0 ]; then
        mkdir -p "${root}"
        root="$(cd "${root}" && pwd)"
    fi
fi

# Run a command, or only print it with --dry-run.
act() {
    if [ "${dry_run}" -eq 1 ]; then
        say "would run: $*"
    else
        "$@"
    fi
}

path_of() {
    printf '%s%s' "${root}" "$1"
}

# --- prerequisites ---------------------------------------------------------------------------

missing_packages=""
unfixable=""

if ! command -v git >/dev/null 2>&1; then
    missing_packages="${missing_packages} git"
fi
if command -v python3 >/dev/null 2>&1; then
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
        found="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        unfixable="python3 >= 3.10 is required (found ${found})"
    fi
    if ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
        missing_packages="${missing_packages} python3-venv"
    fi
else
    missing_packages="${missing_packages} python3 python3-venv"
fi

if [ -n "${missing_packages}" ]; then
    if [ "${install_packages}" -eq 1 ]; then
        if [ "${dry_run}" -eq 0 ]; then
            command -v apt-get >/dev/null 2>&1 || die "apt-get not found; install:${missing_packages}"
        fi
        say "installing prerequisites:${missing_packages}"
        # shellcheck disable=SC2086 # the package list is intentionally split into words
        act apt-get install -y ${missing_packages}
    else
        die "missing prerequisites:${missing_packages}. Install them (for example: apt-get install -y${missing_packages}) or re-run with --install-packages"
    fi
fi
[ -z "${unfixable}" ] || die "${unfixable}"

# --- service user ----------------------------------------------------------------------------

default_service_user() {
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
        echo "${SUDO_USER}"
        return
    fi
    awk -F: '$3 >= 1000 && $3 < 60000 && $1 != "nobody" { print $1; exit }' /etc/passwd 2>/dev/null || true
}

if [ -z "${service_user}" ]; then
    service_user="$(default_service_user)"
fi
[ -n "${service_user}" ] || die "cannot determine the service user; pass --service-user USER"
case "${service_user}" in
*[!a-z0-9_-]* | [0-9-]*) die "invalid service user name: ${service_user}" ;;
esac
id -u "${service_user}" >/dev/null 2>&1 || die "service user does not exist: ${service_user}"
service_group="$(id -gn "${service_user}")"
owner="${service_user}:${service_group}"
say "service user: ${service_user}"

# --- files and directories -------------------------------------------------------------------

ensure_dir() { # path mode
    local dir
    dir="$(path_of "$1")"
    act mkdir -p "${dir}"
    act chmod "$2" "${dir}"
    act chown "${owner}" "${dir}"
}

# Copy a file unless the destination already has identical content.
install_file() { # source destination mode [owner]
    local dest
    dest="$(path_of "$2")"
    if [ -f "${dest}" ] && cmp -s "$1" "${dest}"; then
        say "up to date: $2"
    else
        say "installing: $2"
        act cp "$1" "${dest}"
    fi
    act chmod "$3" "${dest}"
    if [ -n "${4:-}" ]; then
        act chown "$4" "${dest}"
    fi
}

install_unit() { # source file name
    local rendered
    rendered="$(mktemp)"
    sed -e "s|@SERVICE_USER@|${service_user}|g" -e "s|@LTX_ROOT@|${ltx_root}|g" "$1" >"${rendered}"
    install_file "${rendered}" "/etc/systemd/system/$2" 0644
    rm -f "${rendered}"
}

# Create a config file from its example only when it is missing. Either way it ends up owned by
# the service user with mode 0600: the agent's config loader rejects a file readable by group or
# others, and the service (running as that user) must still be able to read it.
install_config() { # example destination [mode]
    local dest mode
    mode="${3:-0600}"
    dest="$(path_of "$2")"
    if [ -e "${dest}" ]; then
        say "keeping existing: $2"
    else
        say "creating from example: $2"
        act cp "$1" "${dest}"
    fi
    act chmod "${mode}" "${dest}"
    act chown "${owner}" "${dest}"
}

ensure_dir /opt/render-agent 0755
ensure_dir /var/lib/render-agent 0750
ensure_dir /etc/render-agent 0755
act mkdir -p "$(path_of /etc/systemd/system)"

install_file "${script_dir}/bootstrap.py" /opt/render-agent/bootstrap.py 0755 "${owner}"
install_unit "${script_dir}/systemd/render-agent.service" render-agent.service
install_unit "${script_dir}/systemd/render-agent-deadman.service" render-agent-deadman.service
install_unit "${script_dir}/systemd/render-agent-deadman.timer" render-agent-deadman.timer
install_config "${script_dir}/agent.env.example" /etc/render-agent/agent.env
install_config "${script_dir}/deploy.env.example" /etc/render-agent/deploy.env
if [ -n "${ltx_root}" ]; then
    install_unit "${script_dir}/systemd/ltx-server.service" ltx-server.service
    install_config "${script_dir}/ltx-server.env.example" /etc/render-agent/ltx-server.env 0644
fi

# --- enabling --------------------------------------------------------------------------------

enable_unit() { # unit wants-directory
    if [ -z "${root}" ]; then
        act systemctl enable "$1"
    else
        # What `systemctl enable` creates for a unit in /etc/systemd/system.
        local wants link
        wants="$(path_of "/etc/systemd/system/$2")"
        link="${wants}/$1"
        if [ -L "${link}" ] && [ "$(readlink "${link}")" = "/etc/systemd/system/$1" ]; then
            say "already enabled: $1"
        else
            act mkdir -p "${wants}"
            act ln -sfn "/etc/systemd/system/$1" "${link}"
        fi
    fi
}

if [ -z "${root}" ]; then
    if ! command -v systemctl >/dev/null 2>&1; then
        if [ "${dry_run}" -eq 0 ]; then
            die "systemctl not found; is this a systemd machine?"
        fi
        say "warning: systemctl not found on this machine (fine for a dry run)"
    fi
    act systemctl daemon-reload
fi
enable_unit render-agent.service multi-user.target.wants
if [ -n "${ltx_root}" ]; then
    enable_unit ltx-server.service multi-user.target.wants
fi
if [ "${enable_deadman}" -eq 1 ]; then
    enable_unit render-agent-deadman.timer timers.target.wants
    say "the dead-man timer is ENABLED: the instance powers off 65 minutes after boot"
else
    say "the dead-man timer is installed but not enabled (use --enable-deadman to opt in)"
fi

say "done. Next: edit /etc/render-agent/agent.env, then see docs/instance-runbook.md"
