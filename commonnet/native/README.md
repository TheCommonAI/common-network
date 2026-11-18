# Native macOS tray

`CommonTray.swift` is the macOS 14+ menu-bar app. It uses standard SwiftUI controls
and automatic layout, with the system popover surface. On macOS 26, Apply uses
SwiftUI's `glassProminent` style. The executable is linked against the SDK named
in `build.json`, rather than inheriting the Python interpreter's AppKit appearance.

The shipped universal executable supports Apple silicon and Intel Macs. The
runtime installs it as `~/.common-network/apps/Common.app`; contributors do not
need Xcode, Swift, or PyObjC on macOS 14+. The Python companion retains the
existing single-instance lock while the native child runs. Windows/Linux keep
the portable controls; older macOS keeps the legacy AppKit fallback.

To change the native source, rebuild on a Mac with Xcode's current SDK:

```sh
python3 commonnet/native/build.py
commonnet/native/bin/CommonTray --self-test
```

Commit the source, universal binary, and updated `build.json` together. The
installer verifies the source and executable hashes before copying the app.
The build is ad-hoc signed; production notarized distribution can replace the
signature without changing the UI architecture.

For a local UI preview, use a disposable config and pass `--show 1` alongside
`--config PATH --status PATH --assets PATH`. Do not point preview service argv at
a real launcher when testing Apply or Leave.
