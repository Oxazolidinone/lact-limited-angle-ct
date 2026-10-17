# shellcheck shell=bash
# SSH settings shared by the scripts that run on the personal machine and talk to a rented Vast machine.
# Not run directly:  source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/remote.sh";  vast_remote <IP> <PORT>
#
# vast_remote sets:
#   host      VAST_USER@IP (VAST_USER defaults to root)
#   port      the SSH port ("Direct ssh connect" entry on the Vast instance card)
#   root      this project's directory on the personal machine
#   remote    the project directory on the rented machine (VAST_ROOT, default /workspace/kltn-lact)
#   ssh_opts  options for ssh / rsync -e, with the key in VAST_KEY when set
vast_remote() {
    host="${VAST_USER:-root}@$1"
    port="$2"
    root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
    remote="${VAST_ROOT:-/workspace/kltn-lact}"
    ssh_opts="-p $port -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o LogLevel=ERROR"
    if [[ -n "${VAST_KEY:-}" ]]; then
        ssh_opts="-i $VAST_KEY $ssh_opts"
    fi
}
