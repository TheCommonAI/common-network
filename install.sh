#!/usr/bin/env bash
# Install the Common Network CLI tools (macOS / Linux).
# The Common Network Alpha (v0.1.5).
#
#   curl -fsSL https://commonai.com.au/install.sh | sh
#
# (commonai.com.au redirects to this file on GitHub — the repository stays the
# single source of truth. The long form works too:
#   curl -fsSL https://raw.githubusercontent.com/TheCommonAI/common-network/main/install.sh | sh )
#
# Installs `common-chat` (talk to the network — needs nothing but Python) and
# `common-join` (contribute a node), installing cloudflared automatically if
# needed.
#
# There used to be a condition on this: common-join was only installed if Ollama
# was already on the machine, because join.py drove Ollama and there was nothing
# it could do without it. That made contributing a machine into installing a
# second product first. join.py now fetches its own model runtime (a ~12MB
# llama.cpp build, see join/runtime.py) and its own weights, so the only thing
# left that a contributor has to have is Python — and cloudflared, which this
# script installs for them.
set -euo pipefail

REPO="TheCommonAI/common-network"
INSTALL_DIR="$HOME/.common-network"
BIN_DIR="$INSTALL_DIR/bin"
RAW="https://raw.githubusercontent.com/$REPO/main"

echo "Installing the Common Network Alpha (v0.1.5)..."
mkdir -p "$BIN_DIR"

OS="$(uname -s)"
ARCH="$(uname -m)"

# --- python3 ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required but wasn't found."
  if [ "$OS" = "Darwin" ]; then
    echo "  Install it with: xcode-select --install"
  else
    echo "  Install it with your package manager, e.g.: sudo apt install python3"
  fi
  exit 1
fi

# --- common-chat (always) ---
echo "Downloading the chat client..."
curl -fsSL "$RAW/chat/chat.py" -o "$INSTALL_DIR/chat.py"
chmod +x "$INSTALL_DIR/chat.py"

cat > "$BIN_DIR/common-chat" <<WRAPPER
#!/usr/bin/env bash
exec python3 "$INSTALL_DIR/chat.py" "\$@"
WRAPPER
chmod +x "$BIN_DIR/common-chat"

# --- common (the unified CLI: ask/join/serve/leave/status/demand/peers/...) ---
echo "Downloading the common CLI..."
curl -fsSL "$RAW/common/common.py" -o "$INSTALL_DIR/common.py"
chmod +x "$INSTALL_DIR/common.py"

cat > "$BIN_DIR/common" <<WRAPPER
#!/usr/bin/env bash
export PATH="$BIN_DIR:\$PATH"
exec python3 "$INSTALL_DIR/common.py" "\$@"
WRAPPER
chmod +x "$BIN_DIR/common"
ln -sf "$BIN_DIR/common" "$BIN_DIR/cmn"

# --- common-join (needs cloudflared; fetches its own runtime and weights) ---
JOIN_INSTALLED=0

# cloudflared
if ! command -v cloudflared >/dev/null 2>&1 && [ ! -x "$BIN_DIR/cloudflared" ]; then
  echo "Installing cloudflared (opens the free tunnel to your machine)..."
  if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    brew install cloudflared
  elif [ "$OS" = "Darwin" ]; then
    case "$ARCH" in
      arm64) CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz" ;;
      *)     CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz" ;;
    esac
    TMP_TGZ="$(mktemp)"
    curl -fsSL "$CF_URL" -o "$TMP_TGZ"
    tar -xzf "$TMP_TGZ" -C "$BIN_DIR"
    rm -f "$TMP_TGZ"
    chmod +x "$BIN_DIR/cloudflared"
  elif [ "$OS" = "Linux" ]; then
    case "$ARCH" in
      x86_64)        CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" ;;
      aarch64|arm64) CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64" ;;
      *) echo "Unsupported architecture: $ARCH. Skipping common-join — install cloudflared manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" ;;
    esac
    if [ -n "${CF_URL:-}" ]; then
      curl -fsSL "$CF_URL" -o "$BIN_DIR/cloudflared"
      chmod +x "$BIN_DIR/cloudflared"
    fi
  fi
fi

if command -v cloudflared >/dev/null 2>&1 || [ -x "$BIN_DIR/cloudflared" ]; then
  echo "Downloading the join script..."
  curl -fsSL "$RAW/join/join.py" -o "$INSTALL_DIR/join.py"
  chmod +x "$INSTALL_DIR/join.py"
  # These two are imported by join.py as siblings, so they are not optional --
  # and because they are imported at first use rather than at the top of
  # join.py, a missing one would not be noticed until the node had already
  # started. runtime.py is the newest of the three: it fetches the llama.cpp
  # runtime and the weights, and machines installed before it existed get it
  # through join.py's own self-update. Fetching it here as well costs one
  # request and saves that round-trip on a fresh install.
  curl -fsSL "$RAW/join/worker.py" -o "$INSTALL_DIR/worker.py"
  curl -fsSL "$RAW/join/runtime.py" -o "$INSTALL_DIR/runtime.py"

  cat > "$BIN_DIR/common-join" <<WRAPPER
#!/usr/bin/env bash
export PATH="$BIN_DIR:\$PATH"
exec python3 "$INSTALL_DIR/join.py" "\$@"
WRAPPER
  chmod +x "$BIN_DIR/common-join"
  JOIN_INSTALLED=1
fi

# --- PATH setup ---
SHELL_RC=""
case "${SHELL:-}" in
  */zsh)  SHELL_RC="$HOME/.zshrc" ;;
  */bash) SHELL_RC="$HOME/.bashrc" ;;
esac

if [ -n "$SHELL_RC" ] && ! grep -qs "$BIN_DIR" "$SHELL_RC" 2>/dev/null; then
  echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$SHELL_RC"
  ADDED_TO_RC=1
fi

echo ""
echo "Done! Open a new terminal, then try:"
echo ""
echo "    common peers         # see the machines on the network"
if [ "$JOIN_INSTALLED" = "1" ]; then
  echo "    common join          # contribute a machine -- then asking works"
  echo ""
  echo "(The first 'common join' downloads a ~15MB model runtime and then the"
  echo " model's weights -- 1.3GB for the smallest in the catalogue, around 4GB"
  echo " for a 7B. Once, then it starts from disk.)"
else
  echo "    (common-join was skipped because cloudflared could not be installed."
  echo "     Install it from the URL above and re-run this installer to"
  echo "     contribute a machine, which is also how asking works.)"
fi
echo ""
echo "(common-chat / common-join still work directly too, if you're used to them)"
echo ""
if [ "${ADDED_TO_RC:-0}" = "1" ]; then
  echo "(Restart your terminal first, or run: export PATH=\"$BIN_DIR:\$PATH\")"
fi
