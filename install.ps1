# Install the Common Network CLI tools (Windows).
# The Common Network (v0.4).
#
#   irm https://commonai.com.au/install.ps1 | iex
#
# (commonai.com.au redirects to this file on GitHub — the repository stays the
# single source of truth. The long form works too:
#   irm https://raw.githubusercontent.com/TheCommonAI/common-network/main/install.ps1 | iex )
#
# Installs `common` (ask / join / status / …), `common-chat`, and `common-join`,
# plus cloudflared if it is missing (other peers reach this machine through a
# Cloudflare quick tunnel). Python is downloaded automatically when needed.
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

Write-Host "Installing the Common Network (v0.4)..."
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

# --- python ---
# A Python that can actually run this, and not merely a Python -- the same
# distinction install.sh draws, for the same reason: common.py annotates with
# `X | None` in 49 places and does not defer them, so it raises on import at
# 3.9 while join.py and runtime.py, which do defer them, import fine.
#
# There are two Windows-specific ways to be wrong here, and both pass a bare
# existence check: `python` may be the App Execution Alias, a stub with no
# interpreter behind it that opens the Store when run, and an old python.org
# install left on PATH answers cheerfully and is too old.
#
# The app's fetched interpreter comes first, so a machine with the desktop app
# has one Python rather than two. Unlike install.sh this is resolved here and
# written into the wrappers as an absolute path: a .cmd file cannot ask an
# interpreter its version without a temporary file per candidate, and Windows
# installs Python per-version rather than swapping one `python3` in place.
$PythonExe = $null
$candidates = @("$InstallDir\runtime\python\python.exe", "$InstallDir\runtime\python\Scripts\python.exe") +
              @(Get-Command python, python3 -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty Source)
foreach ($candidate in $candidates) {
    if (-not $candidate -or -not (Test-Path $candidate)) { continue }
    # Skipped by path, and not by trying it: running the alias is what opens the
    # Store, so probing it to find out what it is would be the pop-up itself.
    if ($candidate -like "*\WindowsApps\*") { continue }
    & $candidate -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $PythonExe = $candidate; break }
}
if (-not $PythonExe) {
    Write-Host "Downloading Common's private Python 3.12..."
    $Uv = "$InstallDir\tools\uv.exe"
    $oldUvInstall = $env:UV_UNMANAGED_INSTALL
    $oldPythonDir = $env:UV_PYTHON_INSTALL_DIR
    $oldUvCache = $env:UV_CACHE_DIR
    try {
        $env:UV_UNMANAGED_INSTALL = "$InstallDir\tools"
        $env:UV_PYTHON_INSTALL_DIR = "$InstallDir\python"
        $env:UV_CACHE_DIR = "$InstallDir\cache\uv"
        if (-not (Test-Path $Uv)) {
            $installer = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString() + ".ps1")
            try {
                Invoke-WebRequest https://astral.sh/uv/install.ps1 -OutFile $installer
                & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer
                if ($LASTEXITCODE -ne 0) { throw "Could not install Common's Python downloader." }
            } finally { Remove-Item $installer -ErrorAction SilentlyContinue }
        }
        & $Uv venv --python 3.12 --managed-python --seed "$InstallDir\runtime\python"
        if ($LASTEXITCODE -ne 0) { throw "Automatic Python setup failed. Check your connection and rerun this installer." }
        $PythonExe = "$InstallDir\runtime\python\Scripts\python.exe"
    } finally {
        $env:UV_UNMANAGED_INSTALL = $oldUvInstall
        $env:UV_PYTHON_INSTALL_DIR = $oldPythonDir
        $env:UV_CACHE_DIR = $oldUvCache
    }
}
Write-Host "Using Python: $PythonExe"

# --- common-chat (always) ---
Write-Host "Downloading the chat client..."
Invoke-WebRequest -Uri "$Raw/chat/chat.py" -OutFile "$InstallDir\chat.py"

$chatWrapper = @"
@echo off
"$PythonExe" -u "$InstallDir\chat.py" %*
"@
Set-Content -Path "$BinDir\common-chat.cmd" -Value $chatWrapper

# --- common (the unified CLI: ask/join/serve/leave/status/demand/peers/...) ---
Write-Host "Downloading the common CLI..."
Invoke-WebRequest -Uri "$Raw/common/common.py" -OutFile "$InstallDir\common.py"

$commonWrapper = @"
@echo off
set PATH=$BinDir;%PATH%
"$PythonExe" -u "$InstallDir\common.py" %*
"@
Set-Content -Path "$BinDir\common.cmd" -Value $commonWrapper
Set-Content -Path "$BinDir\cmn.cmd" -Value $commonWrapper

# --- common-join (needs cloudflared; fetches its own runtime and weights) ---
$joinInstalled = $false

$cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cloudflared -and -not (Test-Path "$BinDir\cloudflared.exe")) {
    Write-Host "Installing cloudflared..."
    $cfUrl = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $cfUrl -OutFile "$BinDir\cloudflared.exe"
}

if ($cloudflared -or (Test-Path "$BinDir\cloudflared.exe")) {
    Write-Host "Downloading the join script..."
    Invoke-WebRequest -Uri "$Raw/join/join.py" -OutFile "$InstallDir\join.py"
    # These two are imported by join.py as siblings, so they are not optional --
    # and because they are imported at first use rather than at the top of
    # join.py, a missing one would not be noticed until the node had already
    # started. runtime.py is the newest of the three: it fetches the llama.cpp
    # runtime and the weights, and machines installed before it existed get it
    # through join.py's own self-update. Fetching it here as well costs one
    # request and saves that round-trip on a fresh install.
    Invoke-WebRequest -Uri "$Raw/join/worker.py" -OutFile "$InstallDir\worker.py"
    Invoke-WebRequest -Uri "$Raw/join/runtime.py" -OutFile "$InstallDir\runtime.py"
    # layers.py runs `common join --network` (the decentralised layer network).
    Invoke-WebRequest -Uri "$Raw/join/layers.py" -OutFile "$InstallDir\layers.py"

    $joinWrapper = @"
@echo off
set PATH=$BinDir;%PATH%
"$PythonExe" -u "$InstallDir\join.py" %*
"@
    Set-Content -Path "$BinDir\common-join.cmd" -Value $joinWrapper
    $joinInstalled = $true
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
Write-Host "    common join --check  # can this machine join? checks only"
if ($joinInstalled) {
    Write-Host "    common join          # serve a slice of the shared model"
    Write-Host "    common ask `"...`"     # ask the network (needs join running)"
    Write-Host ""
    Write-Host "(The first join sets up a private Python environment and downloads"
    Write-Host " only the layers this machine will serve.)"
} else {
    Write-Host "    (common-join was skipped because cloudflared could not be installed."
    Write-Host "     Install it from https://developers.cloudflare.com/cloudflare-one/"
    Write-Host "     connections/connect-networks/downloads/ and re-run this installer:"
    Write-Host "     other peers reach this machine through a Cloudflare tunnel.)"
}
Write-Host ""
