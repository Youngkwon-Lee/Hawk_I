#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/_upstream}"
mkdir -p "$DEST"

clone_at() {
  local repo="$1"
  local sha="$2"
  local name="$3"
  local dir="$DEST/$name"

  if [[ ! -d "$dir/.git" ]]; then
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin "https://github.com/$repo.git"
  fi

  git -C "$dir" fetch --depth 1 origin "$sha"
  git -C "$dir" checkout --detach -q FETCH_HEAD
  local actual
  actual="$(git -C "$dir" rev-parse HEAD)"
  if [[ "$actual" != "$sha" ]]; then
    echo "SHA mismatch for $repo: expected $sha, got $actual" >&2
    exit 2
  fi
  echo "$name $actual"
}

clone_at "Plrbear/PECoP" "af79e55c926457580989b72d27737c6e40f09e8f" "PECoP"
clone_at "yuxumin/CoRe" "f7881a9ff6d8be3cd6549fd58f1456d671537e76" "CoRe"
clone_at "nzl-thu/MUSDL" "205868a826e4df3d0bcac5aa03ca36761aad43cd" "MUSDL"

echo "Pinned upstreams ready under: $DEST"
