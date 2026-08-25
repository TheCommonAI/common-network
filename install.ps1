# Install the Common Network CLI tools (Windows).
#
#   irm https://raw.githubusercontent.com/TheCommonAI/common-network/main/install.ps1 | iex
#
# Always installs `common-chat` (talk to the network — needs nothing but
# Python). Also installs `common-join` (contribute a node) if Ollama is
# present, installing cloudflared automatically if needed.
#
# The wrappers run python with -u (unbuffered). Windows block-buffers Python's
# stdout when it goes through a .cmd wrapper, which puts printed output out of
# order with interactive prompts: the y/n question scrolls above the cursor and
# looks unanswerable. -u makes what you see match what was printed.

$ErrorActionPreference = "Stop"

$Repo = "TheCommonAI/common-network"
$InstallDir = "$env:USERPROFILE\.common-network"
$BinDir = "$InstallDir\bin"
$Raw = "https://raw.githubusercontent.com/$Repo/main"

Write-Host "Installing Common Network CLI tools..."
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

# --- python ---
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python 3 is required but wasn't found."
    Write-Host "Install it from https://www.python.org/downloads/ (check 'Add python.exe to PATH'), then re-run this installer."
    exit 1
}

# --- common-chat (always) ---
Write-Host "Downloading the chat client..."
Invoke-WebRequest -Uri "$Raw/chat/chat.py" -OutFile "$InstallDir\chat.py"

$chatWrapper = @"
@echo off
python -u "$InstallDir\chat.py" %*
"@
Set-Content -Path "$BinDir\common-chat.cmd" -Value $chatWrapper

# --- common (the unified CLI: ask/join/serve/leave/status/demand/peers/...) ---
Write-Host "Downloading the common CLI..."
Invoke-WebRequest -Uri "$Raw/common/common.py" -OutFile "$InstallDir\common.py"

$commonWrapper = @"
@echo off
set PATH=$BinDir;%PATH%
python -u "$InstallDir\common.py" %*
"@
Set-Content -Path "$BinDir\common.cmd" -Value $commonWrapper
Set-Content -Path "$BinDir\cmn.cmd" -Value $commonWrapper

# --- common-join (only if Ollama is present) ---
$joinInstalled = $false
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
    $cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $cloudflared -and -not (Test-Path "$BinDir\cloudflared.exe")) {
        Write-Host "Installing cloudflared..."
        $cfUrl = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        Invoke-WebRequest -Uri $cfUrl -OutFile "$BinDir\cloudflared.exe"
    }

    Write-Host "Downloading the join script..."
    Invoke-WebRequest -Uri "$Raw/join/join.py" -OutFile "$InstallDir\join.py"

    $joinWrapper = @"
@echo off
set PATH=$BinDir;%PATH%
python -u "$InstallDir\join.py" %*
"@
    Set-Content -Path "$BinDir\common-join.cmd" -Value $joinWrapper
    $joinInstalled = $true
} else {
    Write-Host ""
    Write-Host "(Ollama not found -- skipping common-join. Install it from"
    Write-Host " https://ollama.com/download and re-run this installer if you"
    Write-Host " want to contribute a node, not just chat.)"
}

# --- PATH setup ---
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$BinDir", "User")
    Write-Host "Added $BinDir to your PATH."
}

Write-Host ""
Write-Host "Done! Open a new terminal, then try:"
Write-Host ""
Write-Host "    common ask `"hello!`""
Write-Host "    common               # interactive session"
if ($joinInstalled) {
    Write-Host "    common join          # contribute a node"
}
Write-Host ""
