#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# > 1 )); then
    echo "usage: bash scripts/test.sh [test-name-regex]" >&2
    exit 1
fi
bash "$repo_root/scripts/build.sh"
ctest --test-dir "$repo_root/build" --output-on-failure -R "${1:-.*}"
