#!/usr/bin/env bash
# bump-version.sh — Sync version across all Cargo.toml and package.json files
#
# Usage: ./scripts/bump-version.sh 0.2.0
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <new-version>"
  echo "Example: $0 0.2.0"
  exit 1
fi

NEW_VERSION="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Validate semver format
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
  echo "Error: Version must be valid semver (e.g., 0.2.0 or 0.2.0-beta.1)"
  exit 1
fi

echo "Bumping version to $NEW_VERSION"

# --- Cargo.toml files ---
CARGO_FILES=(
  "$ROOT/crabinfer-core/Cargo.toml"
  "$ROOT/crabinfer-server/Cargo.toml"
  "$ROOT/crabinfer-cli/Cargo.toml"
  "$ROOT/crabinfer-node/Cargo.toml"
)

for file in "${CARGO_FILES[@]}"; do
  if [ -f "$file" ]; then
    # Replace the version line in [package] section (first occurrence only)
    sed -i '' "0,/^version = \".*\"/s//version = \"$NEW_VERSION\"/" "$file"
    echo "  Updated $file"
  fi
done

# --- package.json files ---
PKG_FILES=(
  "$ROOT/crabinfer-node/package.json"
  "$ROOT/examples/electron-demo/package.json"
)

for file in "${PKG_FILES[@]}"; do
  if [ -f "$file" ]; then
    # Use node to update JSON properly
    node -e "
      const fs = require('fs');
      const pkg = JSON.parse(fs.readFileSync('$file', 'utf8'));
      pkg.version = '$NEW_VERSION';
      fs.writeFileSync('$file', JSON.stringify(pkg, null, 2) + '\n');
    "
    echo "  Updated $file"
  fi
done

# --- npm platform packages (optionalDependencies in root package.json) ---
PLATFORM_PKGS=(
  "$ROOT/crabinfer-node/npm/darwin-arm64/package.json"
  "$ROOT/crabinfer-node/npm/darwin-x64/package.json"
  "$ROOT/crabinfer-node/npm/linux-x64-gnu/package.json"
  "$ROOT/crabinfer-node/npm/linux-arm64-gnu/package.json"
)

for file in "${PLATFORM_PKGS[@]}"; do
  if [ -f "$file" ]; then
    node -e "
      const fs = require('fs');
      const pkg = JSON.parse(fs.readFileSync('$file', 'utf8'));
      pkg.version = '$NEW_VERSION';
      fs.writeFileSync('$file', JSON.stringify(pkg, null, 2) + '\n');
    "
    echo "  Updated $file"
  fi
done

# Update optionalDependencies versions in root crabinfer-node/package.json
ROOT_PKG="$ROOT/crabinfer-node/package.json"
if [ -f "$ROOT_PKG" ]; then
  node -e "
    const fs = require('fs');
    const pkg = JSON.parse(fs.readFileSync('$ROOT_PKG', 'utf8'));
    if (pkg.optionalDependencies) {
      for (const dep of Object.keys(pkg.optionalDependencies)) {
        pkg.optionalDependencies[dep] = '$NEW_VERSION';
      }
    }
    fs.writeFileSync('$ROOT_PKG', JSON.stringify(pkg, null, 2) + '\n');
  "
  echo "  Updated optionalDependencies in $ROOT_PKG"
fi

# --- Rust version() function (if hardcoded) ---
VERSION_RS="$ROOT/crabinfer-core/src/lib.rs"
if [ -f "$VERSION_RS" ] && grep -q 'fn version()' "$VERSION_RS"; then
  echo "  Note: version() function in lib.rs should read from Cargo.toml at build time"
fi

echo ""
echo "Version bumped to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "  git add -A"
echo "  git commit -m 'chore: bump version to $NEW_VERSION'"
echo "  git tag v$NEW_VERSION"
echo "  git push origin main --tags"
