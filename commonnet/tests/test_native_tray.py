"""The packaged native app must be reproducible and install without Xcode."""
import hashlib
import json
import os
from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest import mock

from commonnet import native_tray


class NativeBundle(unittest.TestCase):
    def make_package(self, root):
        package = root / 'package'
        (package / 'native/bin').mkdir(parents=True)
        (package / 'assets').mkdir()
        source, binary = b'// test source', b'test executable'
        (package / 'native/CommonTray.swift').write_bytes(source)
        (package / 'native/bin/CommonTray').write_bytes(binary)
        (package / 'native/build.json').write_text(json.dumps({
            'source_sha256': hashlib.sha256(source).hexdigest(),
            'binary_sha256': hashlib.sha256(binary).hexdigest(), 'minimum_macos': '14.0'}))
        for name in ('tray-template.svg', 'tray-white.svg'):
            (package / 'assets' / name).write_text('<svg/>')
        return package

    def test_install_uses_packaged_binary_and_real_accessory_app_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = self.make_package(root)
            with mock.patch.object(native_tray.subprocess, 'run') as run:
                executable = native_tray.install_app(package, root / 'Common.app')
                run.assert_not_called()
            self.assertEqual(executable.read_bytes(), b'test executable')
            self.assertTrue(os.access(executable, os.X_OK))
            info = plistlib.loads((root / 'Common.app/Contents/Info.plist').read_bytes())
            self.assertEqual(info['CFBundleIdentifier'], 'network.common.tray')
            self.assertTrue(info['LSUIElement'])
            self.assertTrue((root / 'Common.app/Contents/Resources/tray-template.svg').is_file())

    def test_rejects_incomplete_or_stale_build_before_installing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = self.make_package(root)
            (package / 'native/bin/CommonTray').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                native_tray.install_app(package, root / 'Common.app')
            self.assertFalse((root / 'Common.app').exists())

    def test_shipped_source_and_binary_match_build_manifest(self):
        native = Path(native_tray.__file__).parent / 'native'
        manifest = json.loads((native / 'build.json').read_text())
        self.assertEqual(hashlib.sha256((native / 'CommonTray.swift').read_bytes()).hexdigest(), manifest['source_sha256'])
        self.assertEqual(hashlib.sha256((native / 'bin/CommonTray').read_bytes()).hexdigest(), manifest['binary_sha256'])
        self.assertEqual(manifest['architectures'], ['arm64', 'x86_64'])
