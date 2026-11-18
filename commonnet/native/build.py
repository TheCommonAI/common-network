#!/usr/bin/env python3
"""Rebuild the shipped universal macOS tray with the selected Xcode SDK."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def build():
    root = Path(__file__).resolve().parent
    source = root / 'CommonTray.swift'
    output = root / 'bin' / 'CommonTray'
    output.parent.mkdir(exist_ok=True)
    sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
    with tempfile.TemporaryDirectory(prefix='common-native-') as directory:
        parts = []
        for architecture in ('arm64', 'x86_64'):
            part = str(Path(directory) / architecture)
            subprocess.run(['xcrun', 'swiftc', '-parse-as-library', '-O', '-sdk', sdk,
                            '-target', architecture + '-apple-macos14.0',
                            str(source), '-o', part], check=True)
            parts.append(part)
        subprocess.run(['xcrun', 'lipo', '-create', *parts, '-output', str(output)], check=True)
    subprocess.run(['/usr/bin/codesign', '--force', '--sign', '-', str(output)], check=True)
    manifest = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'sdk': subprocess.check_output(['xcrun', '--show-sdk-version'], text=True).strip(),
                'architectures': ['arm64', 'x86_64'], 'minimum_macos': '14.0'}
    (root / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    build()
