"""Checks before a download, in plain words: is there room on disk and in memory for the layers chosen?"""
from __future__ import annotations

import os
import shutil

GIB = 2**30
HEADROOM_GIB = 1.0  # left free after the download, for the runtime and the OS


class PreflightError(RuntimeError):
    pass


def free_bytes(path: str) -> int:
    """Free space on the drive that holds `path` (or its nearest existing parent)."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return shutil.disk_usage(p).free


def drive_of(path: str) -> str:
    p = os.path.abspath(path)
    drive = os.path.splitdrive(p)[0]
    return drive or p


def check_disk(cache_path: str, need_bytes: float, have_bytes: float, window, free=None) -> None:
    """Raise PreflightError with the exact fix if the rest of the download will not fit."""
    remaining = max(0.0, need_bytes - have_bytes)
    free = free_bytes(cache_path) if free is None else free
    want = remaining + HEADROOM_GIB * GIB
    if free >= want:
        return
    short = (want - free) / GIB
    drive = drive_of(cache_path)
    raise PreflightError(
        f"not enough free disk space for layers {window[0]}-{window[1]}: {remaining / GIB:.2f} GB still to "
        f"download (plus {HEADROOM_GIB:.0f} GB spare), but {drive} has {free / GIB:.2f} GB free.\n"
        f"  Fix one of these, then run the same command again (what is already downloaded is kept):\n"
        f"    - free about {short:.1f} GB on {drive}\n"
        f"    - or keep the download on another drive: set COMMONNET_CACHE=D:\\common-models (any drive with room)\n"
        f"    - or pick fewer layers with --layers (a shorter window needs less space)")


def memory_message(window, need_gib: float, offer_gib: float) -> str:
    return (f"layers {window[0]}-{window[1]} need {need_gib:.2f} GiB of memory here and this machine offers "
            f"{offer_gib:.2f} GiB.\n"
            f"  Fix one of these: close other apps to free memory, or offer more with --stage-gib "
            f"{need_gib:.1f} (if you really have it), or pick a shorter window with --layers.")


def progress(step: int, text: str, total: int = 5) -> str:
    return f"{step}/{total} {text}"
