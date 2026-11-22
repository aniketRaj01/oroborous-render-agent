#!/usr/bin/env bash
# Runs as root from render-agent.service (ExecStopPost with the "+" prefix), after the agent has
# stopped. Powers the instance off only when the agent asked for it: it writes the marker file after
# an idle exit with AUTO_SHUTDOWN_ON_IDLE on. Any other end of the service (crash, signal, deadline,
# a stop from systemctl) leaves no marker, so nothing happens. An OS poweroff makes EC2 STOP the
# instance (its shutdown behaviour is "stop"); no AWS API is called.
#
# Installed to /usr/local/sbin (root-owned) by deploy/install.sh. It must not live anywhere the
# service user can write, and it takes no settings from the environment, because it runs as root.
# The two paths are arguments only so that it can be checked without a real machine.
set -u

marker="${1:-/run/render-agent/poweroff-requested}"
guard="${2:-/etc/render-agent/no-shutdown}"

[ -f "${marker}" ] || exit 0
rm -f "${marker}"

if [ "${SERVICE_RESULT:-}" != "success" ] || [ "${EXIT_STATUS:-}" != "0" ]; then
    echo "poweroff-if-requested: not powering off: the service did not end normally"
    exit 0
fi
if [ -e "${guard}" ]; then
    echo "poweroff-if-requested: not powering off: ${guard} exists"
    exit 0
fi

echo "poweroff-if-requested: idle exit, powering off instance"
if ! /usr/bin/systemctl poweroff; then
    echo "poweroff-if-requested: poweroff failed; the instance stays up"
fi
exit 0
