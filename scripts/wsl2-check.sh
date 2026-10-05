#!/usr/bin/env bash
# hillclimb on Windows, via WSL2: does the sandbox hold? Run inside a WSL2
# Ubuntu shell, with hillclimb installed there (see the "Windows, via WSL2"
# section of docs.hillclimb.sh/security-and-sandboxes). It checks the
# machine, then runs scripts/sandbox-smoke.sh twice: once with the project in
# the Linux filesystem (~/), once on the Windows drive (/mnt/c). Takes about
# two minutes; uses the dummy agent (no login, no tokens). Send back the
# summary at the end.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
smoke="$here/sandbox-smoke.sh"
results=()

echo "### this machine"
echo "kernel:    $(uname -r)"
echo "distro:    $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
echo "python:    $(python3 --version 2>&1)"
echo "bubblewrap: $(bwrap --version 2>/dev/null || echo 'MISSING: sudo apt install -y bubblewrap')"
echo "hillclimb: $(hillclimb --version 2>/dev/null || echo 'MISSING: install it first')"
case "$(uname -r)" in
  *microsoft*WSL2*|*microsoft-standard*) ;;
  *microsoft*) echo "WARNING: this looks like WSL1; the sandbox needs WSL2 (wsl --set-version Ubuntu 2)" ;;
  *) echo "NOTE: not a WSL kernel; this is just a Linux check" ;;
esac
command -v bwrap >/dev/null && command -v hillclimb >/dev/null || { echo "FAIL: install what is MISSING above"; exit 1; }

run_in() {  # label, folder
  echo; echo "=================== $1: $2"
  rm -rf "$2" && mkdir -p "$2" && (cd "$2" && bash "$smoke")
  if [ $? -eq 0 ]; then results+=("PASS  $1"); else results+=("FAIL  $1  (output above)"); fi
  rm -rf "$2"
}

run_in "project in the Linux filesystem" "$HOME/hc-wsl2-check"
# The profile folder, not %USERNAME%: on a Microsoft-account login they differ. cmd.exe runs from /mnt/c because
# a Linux working directory is a UNC path it warns about and cannot use.
winprofile=$(cd /mnt/c 2>/dev/null && cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')
winhome=$([ -n "$winprofile" ] && wslpath -u "$winprofile" 2>/dev/null)
if [ -n "$winhome" ] && [ -d "$winhome" ]; then
  run_in "project on the Windows drive" "$winhome/hc-wsl2-check"
else
  results+=("SKIP  project on the Windows drive (no Windows profile folder found; %USERPROFILE%='${winprofile:-<cmd.exe unavailable>}')")
fi

echo; echo "=================== summary (send this back)"
echo "$(uname -r) | $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") | $(bwrap --version 2>/dev/null) | $(hillclimb --version 2>/dev/null)"
printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q '^FAIL'
