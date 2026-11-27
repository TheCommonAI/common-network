"""Contributor controls: native macOS menu-bar sliders and a portable tray panel.

The service installer owns service.json. This companion only invokes that installer;
resource changes therefore survive a crash or login just like CLI-installed limits.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import secrets
import re
import time
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from commonnet import __version__

DEFAULT_CONFIG = Path.home() / ".common-network" / "service.json"
DEFAULT_STATUS = Path(os.environ.get("COMMONNET_STATUS_FILE") or Path.home() / ".common-network" / "net" / "status.json")
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
        if flag not in ("--permanent", "--foreground"):
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


def pid_is_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) terminates processes on Windows; use a read-only handle instead.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied still means it exists
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def actionable_error(error):
    """Keep useful node messages; turn implementation exceptions into a next step."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(error or "")).strip()
    if not text:
        return "Run common join --check to check this computer, then try joining again."
    if any(part in text for part in ("Traceback (", "returned non-zero exit status", "CalledProcessError", "Command '['")):
        return "The background node could not restart. Try Apply again; if it still fails, check ~/.common-network/join.log."
    text = " ".join(text.split())
    return text[:380] + ("…" if len(text) > 380 else "")


def node_status(path=DEFAULT_STATUS, *, now=None, pid_alive=pid_is_alive):
    """Summarize observed node state without mistaking presence for useful inference."""
    unavailable = {"state": "stopped", "title": "Node not running", "detail": "The node has not reported a running session. Start it with common join --permanent.", "activity": ""}
    try:
        status = json.loads(Path(path).read_text())
        if not isinstance(status, dict):
            return unavailable
    except (OSError, ValueError, TypeError):
        return unavailable
    phase = status.get("state")
    error = actionable_error(status.get("error")) if status.get("error") else ""
    if phase in ("stopped", "leaving") or not pid_alive(status.get("pid")):
        return {**unavailable, "detail": error or "The background service may be restarting. If it stays stopped, run common join --check."}
    stamp = status.get("ts")
    current = time.time() if now is None else now
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or not -5 <= current - stamp <= 15:
        return {"state": "stale", "title": "Node status is out of date", "detail": "The process is running but has stopped reporting status. Wait for recovery, or check ~/.common-network/join.log.", "activity": ""}
    if phase == "error":
        return {"state": "error", "title": "Node needs attention", "detail": error or actionable_error(None), "activity": ""}
    if phase in ("starting", "loading", "downloading", "moving"):
        titles = {"starting": "Starting your node", "loading": "Loading model layers", "downloading": "Downloading model layers", "moving": "Updating hosted layers"}
        detail = "Your node is preparing to contribute. Cached model layers are reused."
        progress = status.get("download")
        if phase == "downloading" and isinstance(progress, dict):
            done, total = progress.get("bytes"), progress.get("total")
            if type(done) in (int, float) and type(total) in (int, float) and total > 0:
                detail = f"{min(100, max(0, round(100 * done / total)))}% downloaded. You can leave this running in the background."
        return {"state": phase, "title": titles[phase], "detail": detail, "activity": ""}
    if phase != "up":
        return {"state": "starting", "title": "Connecting to the network", "detail": "Waiting for the node to report its hosted layers.", "activity": ""}
    window = status.get("window")
    hosting = (f" · hosting layers {window[0]}–{window[1]}" if isinstance(window, list) and len(window) == 2 and all(type(v) is int for v in window) else "")
    stages, coverage = status.get("stages"), status.get("coverage")
    complete = (isinstance(stages, list) and bool(stages) and isinstance(coverage, list)
                and len(stages) == len(coverage) and all(type(count) is int and count > 0 for count in coverage))
    detail = ("Peers currently cover every model layer. The network can attempt complete requests."
              if complete else "Waiting for peers to cover the remaining model layers. Complete requests need every layer online.")
    if error:
        detail += " " + error
    activity = ""
    counts = status.get("activity")
    if isinstance(counts, dict) and type(counts.get("stage_sessions")) is int and counts["stage_sessions"] >= 0:
        count = counts["stage_sessions"]
        seconds = counts.get("window_seconds", 300)
        period = f"last {round(seconds / 60):g} min" if type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0 else "recently"
        activity = (f"{count} model session{'s' if count != 1 else ''} processed · {period} (includes local work)" if count
                    else f"No model sessions processed in the {period}")
    return {"state": "online" if complete else "waiting", "title": "Online" + hosting, "detail": detail, "activity": activity}


@lru_cache(maxsize=1)
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


PRESET_NAMES = ("quiet", "balanced", "maximum", "automatic")
PRESET_DETAILS = {"quiet": "A lighter contribution while you work.",
                  "balanced": "Share spare capacity, with room for your day.",
                  "maximum": "Use the runtime’s available capacity.",
                  "automatic": "Let Common size your contribution."}


def preset_limits(name, memory=None, threads=None):
    """Hardware-relative layer budgets; Maximum retains the runtime's memory reserve."""
    if name not in PRESET_NAMES:
        raise ValueError("Unknown contribution preset")
    if name == "automatic":
        return dict.fromkeys(FLAGS)
    memory = memory_gib() if memory is None else memory
    threads = max(1, os.cpu_count() or 1) if threads is None else max(1, threads)
    if name == "maximum":
        return dict(stage_gib=None, cpu_threads=threads, compute_percent=100, bandwidth_mbps=None)
    fraction, compute, bandwidth = (0.20, 25, 5) if name == "quiet" else (0.35, 50, 20)
    return dict(stage_gib=max(0.5, math.floor(memory * fraction * 2) / 2),
                cpu_threads=max(1, math.floor(threads * (0.25 if name == "quiet" else 0.5))),
                compute_percent=compute, bandwidth_mbps=bandwidth)


def limits_equal(left, right):
    def normalized(value):
        result = {name: value.get(name) for name in FLAGS}
        # An omitted compute budget already means full duty cycle.
        result["compute_percent"] = result["compute_percent"] or 100
        return result
    return normalized(left) == normalized(right)


def matching_preset(limits, memory=None, threads=None):
    for name in PRESET_NAMES:
        if limits_equal(limits, preset_limits(name, memory, threads)):
            return name
    return "custom"


def run_macos(path, status_path=DEFAULT_STATUS):
    import platform
    version = tuple(int(part) for part in platform.mac_ver()[0].split('.')[:2])
    if version >= (14, 0):
        from commonnet.native_tray import run
        return run(path, status_path)
    from commonnet.controls_macos import run_macos as legacy_controls
    return legacy_controls(path, status_path)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Common __COMMON_RELEASE__ · Contribution</title>
<style>
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#fff;color-scheme:dark}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:28px;background:radial-gradient(ellipse at 90% 5%,#a49b91 0,transparent 45%),radial-gradient(ellipse at 10% 90%,#57615a 0,transparent 55%),#252320}.panel{width:420px;max-width:100%;padding:25px;background:linear-gradient(140deg,#302a26c9,#282724bd);backdrop-filter:blur(40px) saturate(130%);border:1px solid #ffffff38;border-radius:30px;box-shadow:inset 0 1px #ffffff29,0 18px 70px #0005}header .release{display:block;margin-top:5px;color:#d7d5d2;font-size:11px;font-weight:600}header{display:flex;justify-content:space-between;align-items:center}.brand{font-size:24px;letter-spacing:-.04em;font-weight:800;display:flex;align-items:center;gap:9px}.brand svg{width:21px;height:21px}.state{font-size:12px;display:flex;align-items:center;gap:8px}.dot{height:8px;width:8px;background:#9b9b9b;border-radius:50%}.dot.online{background:#4bd467}.dot.waiting,.dot.loading,.dot.downloading,.dot.starting{background:#efbb60}.status{padding:18px 0 17px;border-bottom:1px solid #ffffff19;margin-bottom:18px}.status strong{font-size:13px;font-weight:550}.status p{font-size:12px;line-height:1.5;color:#d7d5d2;margin:7px 0}.status small{font-size:11px;color:#bbb9b6}h1{font-size:15px;font-weight:600;margin:0 0 12px}.presets{display:grid;grid-template-columns:1fr 1fr;gap:9px}button{font:inherit;cursor:pointer;color:inherit}button:focus-visible,input:focus-visible{outline:3px solid #69b4ff;outline-offset:3px}.preset{text-align:left;padding:14px;border-radius:16px;border:1px solid #ffffff24;background:linear-gradient(#ffffff0b,#ffffff04);box-shadow:inset 0 1px #ffffff0b;min-height:85px}.preset[aria-pressed=true]{border-color:#57aaff;background:linear-gradient(135deg,#237cce50,#248ef023);box-shadow:inset 0 0 0 1px #5aafff38}.preset .name{display:flex;align-items:center;justify-content:space-between;font-size:14px;font-weight:650}.check{font-size:12px;color:#70baff;visibility:hidden}.preset[aria-pressed=true] .check{visibility:visible}.preset small{display:block;margin-top:7px;font-size:11px;color:#c4c2bf;line-height:1.4}.preset[aria-pressed=true] small{color:#dcecff}.custom{display:block;margin:16px auto 0;border:0;background:none;color:#bcdcff;padding:8px;font-size:12px}#sliders{margin-top:15px;padding-top:4px;border-top:1px solid #ffffff19}.row{margin:17px 0}.line{display:flex;justify-content:space-between;gap:10px;font-size:13px;font-weight:550}output{font-variant-numeric:tabular-nums;color:#d8eaff;font-size:12px}.detail{font-size:10px;color:#bcb9b6;margin:5px 0 10px}input{width:100%;accent-color:#168eff;cursor:pointer}.note{font-size:10px;color:#bcb9b6;line-height:1.5;margin:15px 0}.actions{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-top:14px}#apply{border:1px solid #86c0ff66;background:linear-gradient(#208fff,#086fd6);border-radius:13px;padding:11px 18px;font-size:12px;font-weight:600;box-shadow:inset 0 1px #ffffff38}#apply:disabled{background:#ffffff12;border-color:#ffffff19;color:#ffffff60;box-shadow:none;cursor:default}#stop{border:0;background:none;color:#dcb9b5;font-size:12px;padding:10px 0}button:disabled{cursor:wait;opacity:.6}#message{font-size:11px;color:#c7c4c0;line-height:1.5;min-height:17px;margin-top:13px}[hidden]{display:none!important}@media(prefers-reduced-transparency:reduce){.panel{background:#302e2b;backdrop-filter:none}}@media(prefers-reduced-motion:no-preference){button{transition:background .15s,border-color .15s}}
</style>
<main class="panel"><header><div><span class="brand"><svg aria-hidden="true" width="21" height="21" viewBox="0 0 21 21" fill="none" xmlns="http://www.w3.org/2000/svg">
<line x1="9.01989" y1="3.56359" x2="4.17374" y2="17.2944" stroke-width="2" stroke="white"/>
<circle cx="8.07704" cy="4.03846" r="4.03846" fill="white"/>
<circle cx="16.9616" cy="11.3077" r="4.03846" fill="white"/>
<circle cx="4.03846" cy="16.9615" r="4.03846" fill="white"/>
</svg>
COMMON.</span><span class="release">__COMMON_RELEASE__</span></div><span class="state"><span id="state-label">Checking</span><span class="dot" id="dot"></span></span></header><section class="status" aria-live="polite"><strong id="node-title">Checking your node…</strong><p id="node-detail"></p><small id="node-activity"></small></section><h1>Choose your contribution</h1><div class="presets" role="group" aria-label="Contribution presets"></div><button class="custom" id="custom" aria-expanded="false" aria-controls="sliders">Custom settings ⌄</button><section id="sliders" hidden aria-label="Custom resource limits"></section><p class="note" id="budget-note">Settings apply after your node restarts. Quiet limits may not fit every model.</p><div class="actions"><button id="stop">Leave network</button><button id="apply" disabled>Apply changes</button></div><div id="message" role="status">Your choices are saved across restarts.</div></main>
<script>
const base=location.pathname.replace(/\/$/,''),inputs={},outputs={};let presets={},saved={},working={},busy=false,loaded=false;
const clone=v=>JSON.parse(JSON.stringify(v)),normalized=v=>({...v,compute_percent:v.compute_percent??100}),equal=(a,b)=>['stage_gib','cpu_threads','compute_percent','bandwidth_mbps'].every(k=>(normalized(a)[k]??null)===(normalized(b)[k]??null));
const definitions=[['stage_gib','Model memory','RAM or GPU memory for model layers',.5],['cpu_threads','CPU threads','Threads available to the model runtime',1],['compute_percent','Compute share','Paces served inference on CPU or GPU',1],['bandwidth_mbps','Bandwidth','Shared download and peer payload budget',1]];
const format=(name,value)=>value==null?(name==='bandwidth_mbps'?'Unlimited':'Automatic'):value+({stage_gib:' GiB',cpu_threads:' threads',compute_percent:'%',bandwidth_mbps:' Mbps'}[name]);
function redraw(){let selected=Object.keys(presets).find(k=>equal(working,presets[k].limits));document.querySelectorAll('.preset').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.preset===selected)));document.querySelector('#apply').disabled=busy||!loaded||equal(saved,working);document.querySelector('#stop').disabled=busy||!loaded;document.querySelector('#custom').disabled=busy||!loaded;for(const[k,input]of Object.entries(inputs)){input.disabled=busy;input.value=working[k]??(k==='compute_percent'?100:0);outputs[k].textContent=format(k,working[k]);}}
async function refreshStatus(){try{const r=await fetch(base+'/status');if(!r.ok)throw Error();const s=await r.json();document.querySelector('#node-title').textContent=s.title;document.querySelector('#node-detail').textContent=s.detail;document.querySelector('#node-activity').textContent=s.activity;document.querySelector('#dot').className='dot '+s.state;document.querySelector('#state-label').textContent=({online:'Online',waiting:'Online',stopped:'Offline',stale:'Check node',error:'Attention'})[s.state]??'Joining';}catch(e){document.querySelector('#state-label').textContent='Unavailable';document.querySelector('#dot').className='dot';}}
refreshStatus();setInterval(refreshStatus,2000);
fetch(base+'/config').then(r=>{if(!r.ok)throw Error();return r.json()}).then(config=>{presets=config.presets;saved=clone(config.limits);working=clone(saved);for(const[name,preset]of Object.entries(presets)){const button=document.createElement('button');button.className='preset';button.dataset.preset=name;button.setAttribute('aria-pressed','false');const label=document.createElement('span');label.className='name';label.textContent=name[0].toUpperCase()+name.slice(1);const check=document.createElement('span');check.className='check';check.textContent='✓';check.setAttribute('aria-hidden','true');label.append(check);const detail=document.createElement('small');detail.textContent=preset.detail;button.append(label,detail);button.onclick=()=>{if(busy)return;working=clone(preset.limits);redraw();document.querySelector('#message').textContent='Apply changes to use '+label.firstChild.textContent+'.'};document.querySelector('.presets').append(button);}
for(const[name,title,detail,step]of definitions){const row=document.createElement('div');row.className='row';const line=document.createElement('div');line.className='line';const label=document.createElement('label');label.htmlFor=name;label.textContent=title;const output=document.createElement('output');output.htmlFor=name;line.append(label,output);const desc=document.createElement('div');desc.className='detail';desc.textContent=detail;const input=document.createElement('input');input.type='range';input.id=name;input.min=name==='compute_percent'?1:0;input.max=config.maximums[name];input.step=step;input.oninput=()=>{working[name]=Number(input.value)||null;redraw();document.querySelector('#message').textContent='Custom limits · apply to restart your node.'};row.append(line,desc,input);document.querySelector('#sliders').append(row);inputs[name]=input;outputs[name]=output;}
loaded=true;redraw();}).catch(()=>{document.querySelector('#message').textContent='Could not load saved settings. Reopen Common controls.';redraw();});
document.querySelector('#custom').onclick=()=>{const button=document.querySelector('#custom'),open=button.getAttribute('aria-expanded')!=='true';button.setAttribute('aria-expanded',String(open));document.querySelector('#sliders').hidden=!open;button.textContent=open?'Hide custom settings ⌃':'Custom settings ⌄';document.querySelector('#budget-note').textContent=open?'Memory is a model-layer budget, not a total RAM cap. Compute and bandwidth are cooperative limits. Applying restarts your node.':'Settings apply after your node restarts. Quiet limits may not fit every model.';};
async function change(stop){if(busy||!loaded)return;if(stop&&!confirm('Leave the network and remove automatic startup?'))return;busy=true;redraw();document.querySelectorAll('.preset').forEach(b=>b.disabled=true);const message=document.querySelector('#message');message.textContent=stop?'Leaving the network…':'Applying changes and reconnecting…';try{const r=await fetch(base+(stop?'/stop':'/apply'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(working)});const result=await r.json();if(!r.ok)throw Error(result.error);if(!stop)saved=clone(working);message.textContent=result.message;}catch(e){message.textContent=e.message;}finally{busy=false;redraw();document.querySelectorAll('.preset').forEach(b=>b.disabled=false);}}
document.querySelector('#apply').onclick=()=>change(false);document.querySelector('#stop').onclick=()=>change(true);
</script></html>
'''


def make_server(path, status_path=DEFAULT_STATUS):
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
                return self.reply(200, HTML.replace("__COMMON_RELEASE__", f"v{__version__} Beta"), "text/html; charset=utf-8")
            if self.path == base + "/status":
                return self.reply(200, node_status(status_path))
            if self.path == base + "/config":
                try:
                    config = load_config(path)
                    limits = config.get("limits", {})
                    return self.reply(200, {"limits": limits, "presets": {name: {"limits": preset_limits(name), "detail": PRESET_DETAILS[name]} for name in PRESET_NAMES}, "selected": matching_preset(limits), "maximums": {
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
                self.reply(400, {"error": actionable_error(error)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    return server, f"http://127.0.0.1:{server.server_port}{base}/"


def run_portable(path):
    server, url = make_server(path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import pystray
        from PIL import Image
        image = Image.open(Path(__file__).with_name("assets") / "tray-white.png").convert("RGBA")
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
