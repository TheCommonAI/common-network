#!/usr/bin/env bash
# Install the Common Network CLI tools (macOS / Linux).
#
#   curl -fsSL https://raw.githubusercontent.com/TheCommonAI/common-network/main/install.sh | sh
#
# Always installs `common-chat` (talk to the network — needs nothing but
# Python). Also installs `common-join` (contribute a node) if Ollama is
# present, installing cloudflared automatically if needed.
set -euo pipefail

REPO="TheCommonAI/common-network"
INSTALL_DIR="$HOME/.common-network"
BIN_DIR="$INSTALL_DIR/bin"
RAW="https://raw.githubusercontent.com/$REPO/main"

echo "Installing Common Network CLI tools..."
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

# --- common-join (only if Ollama is present) ---
JOIN_INSTALLED=0
if command -v ollama >/dev/null 2>&1; then
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

    cat > "$BIN_DIR/common-join" <<WRAPPER
#!/usr/bin/env bash
export PATH="$BIN_DIR:\$PATH"
exec python3 "$INSTALL_DIR/join.py" "\$@"
WRAPPER
    chmod +x "$BIN_DIR/common-join"
    JOIN_INSTALLED=1
  fi
else
  echo ""
  echo "(Ollama not found — skipping common-join. Install it from"
  echo " https://ollama.com/download and re-run this installer if you"
  echo " want to contribute a node, not just chat.)"
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
echo "Done! Try:"
echo ""
echo "    common ask \"hello!\""
echo "    common               # interactive session"
if [ "$JOIN_INSTALLED" = "1" ]; then
  echo "    common join          # contribute a node"
fi
echo ""
echo "(common-chat / common-join still work directly too, if you're used to them)"
echo ""
if [ "${ADDED_TO_RC:-0}" = "1" ]; then
  echo "(Restart your terminal first, or run: export PATH=\"$BIN_DIR:\$PATH\")"
fi
