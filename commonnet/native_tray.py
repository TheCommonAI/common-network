"""Install and run the compiled SwiftUI menu-bar app; no compiler needed at runtime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess


BUNDLE_ID = 'network.common.tray'


def install_app(source=None, destination=None):
    package = Path(source) if source else Path(__file__).resolve().parent
    native = package / 'native'
    manifest = json.loads((native / 'build.json').read_text())
    binary = native / 'bin' / 'CommonTray'
    if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest['binary_sha256']:
        raise RuntimeError('The native tray download is incomplete. Update Common and try again.')
    if hashlib.sha256((native / 'CommonTray.swift').read_bytes()).hexdigest() != manifest['source_sha256']:
        raise RuntimeError('The native tray source and build do not match. Update Common and try again.')
    app = Path(destination) if destination else Path.home() / '.common-network' / 'apps' / 'Common.app'
    executable = app / 'Contents' / 'MacOS' / 'CommonTray'
    resources = app / 'Contents' / 'Resources'
    executable.parent.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    staged = executable.with_suffix('.new')
    shutil.copyfile(binary, staged)
    staged.chmod(0o755)
    os.replace(staged, executable)
    for name in ('tray-template.svg', 'tray-white.svg'):
        shutil.copyfile(package / 'assets' / name, resources / name)
    info = {'CFBundleIdentifier': BUNDLE_ID, 'CFBundleName': 'Common',
            'CFBundleDisplayName': 'Common', 'CFBundleExecutable': 'CommonTray',
            'CFBundlePackageType': 'APPL', 'CFBundleInfoDictionaryVersion': '6.0',
            'CFBundleShortVersionString': '0.4.0', 'CFBundleVersion': '1',
            'CommonBuildSource': manifest['source_sha256'],
            'LSUIElement': True, 'LSMinimumSystemVersion': manifest['minimum_macos'],
            'NSHighResolutionCapable': True}
    with (app / 'Contents' / 'Info.plist').open('wb') as file:
        plistlib.dump(info, file)
    return executable


def run(path, status_path):
    executable = install_app()
    result = subprocess.run([str(executable), '--config', str(path), '--status', str(status_path),
                             '--assets', str(executable.parent.parent / 'Resources')])
    if result.returncode:
        raise RuntimeError(f'Common’s native tray exited with code {result.returncode}. Reopen Common controls.')
