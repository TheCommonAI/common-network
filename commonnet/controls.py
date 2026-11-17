"""Contributor controls: native macOS menu-bar sliders and a portable tray panel.

The service installer owns service.json. This companion only invokes that installer;
resource changes therefore survive a crash or login just like CLI-installed limits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_CONFIG = Path.home() / ".common-network" / "service.json"
FLAGS = {"stage_gib": "--stage-gib", "cpu_threads": "--cpu-threads",
         "compute_percent": "--compute-percent", "bandwidth_mbps": "--bandwidth-mbps"}
RESOURCE_FLAGS = {*FLAGS.values(), "--memory-gib"}


def load_config(path):
    config = json.loads(Path(path).read_text())
    argv = config.get("argv")
    if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(v, str) for v in argv):
        raise ValueError("The background service configuration has no valid launch command. Run common join --permanent again.")
    return config


def validate_limits(limits):
    if not isinstance(limits, dict) or set(limits) - set(FLAGS):
        raise ValueError("Unknown resource setting")
    result = {}
    for name in FLAGS:
        value = limits.get(name)
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive, or automatic")
            if name == "compute_percent" and not 1 <= value <= 100:
                raise ValueError("Compute share must be between 1 and 100 percent")
            if name == "cpu_threads" and (value != int(value) or value > (os.cpu_count() or 1)):
                raise ValueError("CPU threads must be a whole number available on this computer")
        result[name] = value
    return result


def apply_command(config, limits):
    limits = validate_limits(limits)
    old = config["argv"]
    argv = list(old[:2])
    index = 2
    while index < len(old):
        argument = old[index]
        flag = argument.split("=", 1)[0]
        if flag in RESOURCE_FLAGS:
            index += 1 if "=" in argument else 2
            continue
        if flag != "--permanent":
            argv.append(argument)
        index += 1
    for name, value in limits.items():
        if value is not None:
            argv.append(f"{FLAGS[name]}={int(value) if name == 'cpu_threads' else value}")
    return argv + ["--permanent"]


def change_service(path, limits=None, *, stop=False, run=subprocess.run):
    config = load_config(path)
    command = (config["argv"][:2] + ["--no-update", "--remove-permanent"] if stop
               else apply_command(config, limits))
    result = run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        # Show the actionable tail, without sending logs to a remote service.
        raise RuntimeError((result.stderr or result.stdout or "The service could not be changed.")[-1500:])


def memory_gib():
    try:
        if sys.platform == "darwin":
            size = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        else:
            size = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return max(1, int(size / 2**30))
    except (OSError, ValueError, AttributeError):
        return 64


def display_value(name, value):
    if value is None or value == 0:
        return "Unlimited" if name == "bandwidth_mbps" else "Automatic"
    if name == "stage_gib":
        return f"{value:g} GiB"
    if name == "cpu_threads":
        return f"{int(value)} threads"
    if name == "compute_percent":
        return f"{int(value)}%"
    return f"{value:g} Mbps"


SLIDERS = (("stage_gib", "Model memory", "RAM or GPU memory offered for model layers"),
           ("cpu_threads", "CPU threads", "Threads available to the model runtime"),
           ("compute_percent", "Compute share", "Paces served inference on the CPU or GPU"),
           ("bandwidth_mbps", "Bandwidth", "Shared download and peer payload budget"))


def run_macos(path):
    import AppKit as A
    import objc
    from Foundation import NSObject, NSTimer

    class Controls(NSObject):
        def applicationDidFinishLaunching_(self, notification):
            self.config = load_config(path)
            self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(2.0, self, "checkConfig:", None, True)
            self.sliders, self.readouts = {}, {}
            self.item = A.NSStatusBar.systemStatusBar().statusItemWithLength_(A.NSVariableStatusItemLength)
            button = self.item.button()
            image = A.NSImage.imageWithSystemSymbolName_accessibilityDescription_("circle.hexagongrid.fill", "Common resource controls")
            if image:
                image.setTemplate_(True)
                button.setImage_(image)
            else:
                button.setTitle_("Common")
            button.setToolTip_("Common — choose the resources you share")
            button.setTarget_(self)
            button.setAction_("toggle:")
            controller = A.NSViewController.alloc().init()
            view = A.NSVisualEffectView.alloc().initWithFrame_(((0, 0), (400, 604)))
            view.setMaterial_(A.NSVisualEffectMaterialPopover)
            view.setState_(A.NSVisualEffectStateActive)
            controller.setView_(view)
            self.popover = A.NSPopover.alloc().init()
            self.popover.setContentViewController_(controller)
            self.popover.setBehavior_(A.NSPopoverBehaviorTransient)
            self.popover.setContentSize_((400, 604))

            def label(text, x, y, width, height, size=13, bold=False, secondary=False):
                item = A.NSTextField.labelWithString_(text)
                item.setFrame_(((x, y), (width, height)))
                item.setFont_((A.NSFont.boldSystemFontOfSize_ if bold else A.NSFont.systemFontOfSize_)(size))
                if secondary:
                    item.setTextColor_(A.NSColor.labelColor().colorWithAlphaComponent_(0.72))
                view.addSubview_(item)
                return item

            label("Common", 24, 551, 260, 32, 24, True)
            label("A little spare power. A shared network.", 24, 525, 352, 24, secondary=True)
            limits = self.config.get("limits", {})
            maximums = {"stage_gib": max(memory_gib(), limits.get("stage_gib") or 0),
                        "cpu_threads": os.cpu_count() or 1, "compute_percent": 100,
                        "bandwidth_mbps": max(1000, limits.get("bandwidth_mbps") or 0)}
            for index, (name, title, detail) in enumerate(SLIDERS):
                y = 477 - index * 92
                label(title, 24, y, 210, 22, 14, True)
                readout = label("", 234, y, 142, 22, 13)
                readout.setAlignment_(A.NSTextAlignmentRight)
                label(detail, 24, y - 21, 352, 20, 11, secondary=True)
                slider = A.NSSlider.alloc().initWithFrame_(((24, y - 51), (352, 24)))
                slider.setMinValue_(1 if name == "compute_percent" else 0)
                slider.setMaxValue_(maximums[name])
                slider.setDoubleValue_(limits.get(name) or (100 if name == "compute_percent" else 0))
                slider.setContinuous_(True)
                slider.setTarget_(self)
                slider.setAction_("slide:")
                view.addSubview_(slider)
                self.sliders[name], self.readouts[name] = slider, readout
            label("Memory is a model-layer budget, not a total RAM cap.", 24, 124, 352, 20, 11, secondary=True)
            label("Applying restarts your node. Other peers may take over.", 24, 104, 352, 20, 11, secondary=True)
            self.message = label("Your choices are saved across restarts.", 24, 78, 352, 20, 11, secondary=True)
            self.apply = A.NSButton.buttonWithTitle_target_action_("Apply limits", self, "apply:")
            self.apply.setFrame_(((240, 25), (136, 36)))
            self.apply.setBezelStyle_(A.NSBezelStyleRounded)
            self.apply.setKeyEquivalent_("\r")
            view.addSubview_(self.apply)
            self.stop = A.NSButton.buttonWithTitle_target_action_("Stop contributing", self, "stop:")
            self.stop.setFrame_(((20, 25), (180, 36)))
            self.stop.setBezelStyle_(A.NSBezelStyleRounded)
            view.addSubview_(self.stop)
            self.slide_(None)

        def checkConfig_(self, timer):
            if not path.exists():
                A.NSApplication.sharedApplication().terminate_(None)

        def toggle_(self, sender):
            if self.popover.isShown():
                self.popover.performClose_(sender)
            else:
                try:
                    self.config = load_config(path)
                    for name, slider in self.sliders.items():
                        value = self.config.get("limits", {}).get(name)
                        slider.setMaxValue_(max(slider.maxValue(), value or 0))
                        slider.setDoubleValue_(value or (100 if name == "compute_percent" else 0))
                    self.slide_(None)
                except (OSError, ValueError):
                    self.message.setStringValue_("Could not reload your saved limits.")
                self.popover.showRelativeToRect_ofView_preferredEdge_(self.item.button().bounds(), self.item.button(), A.NSMinYEdge)
                A.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

        @objc.python_method
        def values(self):
            values = {}
            for name, slider in self.sliders.items():
                value = slider.doubleValue()
                value = round(value * 2) / 2 if name == "stage_gib" else round(value)
                values[name] = value or None
            return values

        def slide_(self, sender):
            for name, value in self.values().items():
                self.readouts[name].setStringValue_(display_value(name, value))

        def apply_(self, sender):
            self.begin(False)

        def stop_(self, sender):
            self.begin(True)

        @objc.python_method
        def begin(self, stop):
            self.apply.setEnabled_(False)
            self.stop.setEnabled_(False)
            self.message.setStringValue_("Stopping…" if stop else "Saving limits and reconnecting…")
            values = self.values()
            def work():
                try:
                    change_service(path, values, stop=stop)
                    message = "Contribution stopped." if stop else "Saved. Your node is reconnecting."
                except Exception as error:
                    message = str(error)
                self.performSelectorOnMainThread_withObject_waitUntilDone_("finished:", message, False)
            threading.Thread(target=work, daemon=True).start()

        def finished_(self, message):
            self.apply.setEnabled_(True)
            self.stop.setEnabled_(True)
            self.message.setStringValue_(message.splitlines()[-1][:110])
            self.message.setToolTip_(message)

    app = A.NSApplication.sharedApplication()
    app.setActivationPolicy_(A.NSApplicationActivationPolicyAccessory)
    delegate = Controls.alloc().init()
    app.setDelegate_(delegate)
    app.run()


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Common · Your contribution</title>
<style>
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#193b31;background:#f2f5ef}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px}.panel{width:440px;max-width:100%;background:#fffef9;padding:32px;border:1px solid #dde5d8;border-radius:24px;box-shadow:0 18px 70px #24422b12}.brand{font-size:12px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;display:flex;align-items:center;gap:9px}.dot{height:9px;width:9px;background:#4b8c67;border-radius:50%}h1{font-size:30px;letter-spacing:-.04em;margin:20px 0 8px}p{color:#63746b;font-size:13px;line-height:1.5;margin:0 0 24px}.row{margin:27px 0}.line{display:flex;justify-content:space-between;gap:10px;font-size:14px;font-weight:650}output{font-variant-numeric:tabular-nums;color:#477458;font-weight:500}.detail{font-size:11px;color:#728078;margin:7px 0 12px}input{width:100%;accent-color:#397453;cursor:pointer}button{font:inherit;font-size:13px;border:0;border-radius:10px;padding:12px 16px;cursor:pointer}button:disabled{opacity:.5;cursor:wait}.actions{display:flex;justify-content:space-between;gap:12px;border-top:1px solid #e4e9df;padding-top:20px}#apply{background:#285a40;color:white}#stop{background:#edf1e9;color:#4b5d50}#message{font-size:12px;color:#55705d;min-height:35px;white-space:pre-wrap;margin-bottom:16px}.note{font-size:11px;margin-top:25px}
</style><main class="panel"><div class="brand"><span class="dot"></span>Common</div><h1>Your spare power.<br>Our shared network.</h1><p>Choose how much this computer contributes.</p><div id="sliders"></div><p class="note">Memory is a model-layer budget, not a total RAM cap. Compute share paces served inference; bandwidth paces downloads and peer payloads. Applying restarts your node.</p><div id="message" role="status">Your choices are saved across restarts.</div><div class="actions"><button id="stop">Stop contributing</button><button id="apply">Apply limits</button></div></main>
<script>
const base=location.pathname.replace(/\/$/,''),inputs={};
const definitions=[['stage_gib','Model memory','RAM or GPU memory offered for model layers',0.5],['cpu_threads','CPU threads','Threads available to the model runtime',1],['compute_percent','Compute share','Paces served inference on the CPU or GPU',1],['bandwidth_mbps','Bandwidth','Shared download and peer payload budget',1]];
const format=(name,value)=>!value?(name==='bandwidth_mbps'?'Unlimited':'Automatic'):value+({stage_gib:' GiB',cpu_threads:' threads',compute_percent:'%',bandwidth_mbps:' Mbps'}[name]);
fetch(base+'/config').then(r=>r.json()).then(config=>{
for(const [name,title,detail,step] of definitions){const row=document.createElement('div');row.className='row';const line=document.createElement('div');line.className='line';const label=document.createElement('label');label.htmlFor=name;label.textContent=title;const output=document.createElement('output');output.htmlFor=name;line.append(label,output);const desc=document.createElement('div');desc.className='detail';desc.textContent=detail;const input=document.createElement('input');input.type='range';input.id=name;input.min=name==='compute_percent'?1:0;input.max=config.maximums[name];input.step=step;input.value=config.limits[name]??(name==='compute_percent'?100:0);input.oninput=()=>output.textContent=format(name,Number(input.value));input.oninput();row.append(line,desc,input);document.querySelector('#sliders').append(row);inputs[name]=input;}
}).catch(()=>document.querySelector('#message').textContent='Could not load the background service. Run common join --permanent again.');
async function change(stop){const buttons=document.querySelectorAll('button');buttons.forEach(b=>b.disabled=true);const message=document.querySelector('#message');message.textContent=stop?'Stopping…':'Saving limits and reconnecting…';try{const limits=Object.fromEntries(Object.entries(inputs).map(([k,v])=>[k,Number(v.value)||null]));const r=await fetch(base+(stop?'/stop':'/apply'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(limits)});const result=await r.json();if(!r.ok)throw Error(result.error);message.textContent=result.message;}catch(e){message.textContent=e.message;}finally{buttons.forEach(b=>b.disabled=false);}}
document.querySelector('#apply').onclick=()=>change(false);document.querySelector('#stop').onclick=()=>change(true);
</script></html>'''


def make_server(path):
    token = secrets.token_urlsafe(32)
    base = "/" + token
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass

        def reply(self, status, payload, content_type="application/json"):
            body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.allowed():
                return self.reply(403, {"error": "Invalid host"})
            if self.path in (base, base + "/"):
                return self.reply(200, HTML, "text/html; charset=utf-8")
            if self.path == base + "/config":
                try:
                    config = load_config(path)
                    limits = config.get("limits", {})
                    return self.reply(200, {"limits": limits, "maximums": {
                        "stage_gib": max(memory_gib(), limits.get("stage_gib") or 0),
                        "cpu_threads": os.cpu_count() or 1, "compute_percent": 100,
                        "bandwidth_mbps": max(1000, limits.get("bandwidth_mbps") or 0)}})
                except Exception as error:
                    return self.reply(500, {"error": str(error)})
            self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.allowed() or self.path not in (base + "/apply", base + "/stop"):
                return self.reply(403, {"error": "Invalid control request"})
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
                return self.reply(403, {"error": "Invalid origin"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid settings size")
                limits = validate_limits(json.loads(self.rfile.read(length)))
                with lock:
                    stop = self.path.endswith("/stop")
                    change_service(path, limits, stop=stop)
                self.reply(200, {"message": "Contribution stopped." if stop else "Saved. Your node is reconnecting."})
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                self.reply(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    return server, f"http://127.0.0.1:{server.server_port}{base}/"


def run_portable(path):
    server, url = make_server(path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import pystray
        from PIL import Image, ImageDraw
        image = Image.new("RGBA", (64, 64))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((4, 4, 60, 60), radius=17, fill="#285a40")
        draw.arc((17, 17, 47, 47), 45, 315, fill="white", width=7)
        icon = pystray.Icon("Common", image, "Common resource controls", menu=pystray.Menu(
            pystray.MenuItem("Resource controls…", lambda *_: webbrowser.open(url), default=True),
            pystray.MenuItem("Hide icon", lambda *_: icon.stop())))
        def watch():
            while path.exists():
                if stopped.wait(2):
                    return
            icon.stop()
        stopped = threading.Event()
        threading.Thread(target=watch, daemon=True).start()
        try:
            icon.run()
        finally:
            stopped.set()
    except (ImportError, RuntimeError) as error:
        # A desktop without a tray still gets the same sliders in its browser.
        print(f"Tray unavailable ({error}); opening Common controls in your browser.", file=sys.stderr)
        webbrowser.open(url)
        try:
            while path.exists():
                threading.Event().wait(2)
        except KeyboardInterrupt:
            pass
    finally:
        server.shutdown()
        server.server_close()


def acquire_instance(path):
    """Hold an OS lock for this config; stale lock files are harmless after a crash."""
    handle = path.with_name(path.name + ".controls.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, 2) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    instance = None
    try:
        load_config(args.config)
        instance = acquire_instance(args.config)
        if instance is None:
            return 0
        if sys.platform == "darwin":
            run_macos(args.config)
        else:
            run_portable(args.config)
    except (OSError, ValueError, ImportError) as error:
        print(f"Common controls: {error}", file=sys.stderr)
        return 1
    finally:
        if instance is not None:
            instance.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
