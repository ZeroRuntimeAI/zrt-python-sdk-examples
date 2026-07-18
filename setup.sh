#!/usr/bin/env bash
#
# Setup for the Zero Runtime Python SDK examples.
#
# Creates an isolated environment, installs the SDK plus example dependencies,
# and seeds a local .env from the template. Prefers `uv` (fast, lockfile-driven)
# and falls back to a plain venv + pip. Safe to re-run: an existing .env is never
# overwritten.
set -euo pipefail

# Resolve the repo root so the script works from any working directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -t 1 ]; then
  C_INFO=$'\033[0;36m'; C_WARN=$'\033[0;33m'; C_ERR=$'\033[0;31m'; C_OFF=$'\033[0m'
else
  C_INFO=''; C_WARN=''; C_ERR=''; C_OFF=''
fi
log()  { printf '%s[setup]%s %s\n' "$C_INFO" "$C_OFF" "$*"; }
warn() { printf '%s[setup] %s%s\n' "$C_WARN" "$*" "$C_OFF" >&2; }
die()  { printf '%s[setup] error:%s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

[ -f pyproject.toml ] || die "pyproject.toml not found — run this from the examples repo root."

if command -v uv >/dev/null 2>&1; then
  log "uv detected → uv sync"
  uv sync
  run_hint="uv run features/basic_cascade.py"
else
  warn "uv not found — falling back to venv + pip (install uv for faster setup: https://docs.astral.sh/uv/)."
  command -v python3 >/dev/null 2>&1 || die "python3 not found. Install Python 3.11+ and re-run."

  if [ ! -d .venv ]; then
    log "creating virtualenv at .venv"
    python3 -m venv .venv
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate

  log "upgrading pip"
  python -m pip install --quiet --upgrade pip

  log "installing zrt"
  pip install --quiet "zrt"

  log "installing example dependencies"
  pip install --quiet "python-dotenv>=1.0"

  run_hint="source .venv/bin/activate && python features/basic_cascade.py"
fi

# Seed .env from the template without clobbering an existing one.
if [ -f .env ]; then
  log ".env already exists — left untouched."
elif [ -f .env.example ]; then
  cp .env.example .env
  log "created .env from .env.example"
else
  warn ".env.example not found — skipping .env creation."
fi

cat <<EOF

Setup complete. Next:
  1. Edit .env — set ZRT_AUTH_TOKEN and the provider keys your example needs.
  2. Run an example:
       $run_hint
EOF