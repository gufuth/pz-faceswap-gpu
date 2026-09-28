"""pz_faceswap_runpod_runner.py — runs ON the rented pod (stdlib only).

Uploaded by dispatch_runpod_faceswap.py and executed over SSH. Mirrors the
local FaceFusion invocation in PROJECT ZERO/pz-app/pz_faceswap.py's
build_swap_cmd() flag-for-flag, so a swap made on rented hardware and one made
on the local 4070 pass FaceFusion the same parameters — the only thing that
changes is which GPU runs it.

Bootstrap (skipped on the pre-baked ghcr.io/gufuth/pz-faceswap-gpu image, which
carries all of this plus the model weights at /opt/facefusion; otherwise): clone
FaceFusion 3.6.0, create a venv, install its requirements + onnxruntime-gpu.
FaceFusion's own headless-run auto-downloads whichever model weights the run
actually needs on first use — this runner does not pre-fetch them separately.

Always prints exactly one JSON object on the last stdout line.
"""

import argparse
import re
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path("/workspace")
# Pre-baked image (ghcr.io/gufuth/pz-faceswap-gpu): FaceFusion 3.6.0, its venv
# and every model the Player can ask for already live at /opt/facefusion, so the
# clone/venv/pip bootstrap below is skipped. On the bare pytorch image that dir
# does not exist and the runner falls back to installing into /workspace.
BAKED_FF_DIR = Path(os.environ.get("PZ_BAKED_FF_DIR", "/opt/facefusion"))
if (BAKED_FF_DIR / "venv" / "bin" / "python").is_file():
    FF_DIR = BAKED_FF_DIR
else:
    FF_DIR = WORKSPACE / "facefusion"
VENV_PY = FF_DIR / "venv" / "bin" / "python"


def _emit(obj, code=0):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()
    sys.exit(code)


def _run(cmd, cwd=None, timeout=1200):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def ensure_ffmpeg():
    """FaceFusion shells its own ffmpeg for encode/mux — the base pytorch
    image doesn't ship it. Caught live: a full clone+venv+pip bootstrap
    (all rc=0) still failed at headless-run with '[FACEFUSION.CORE] ffmpeg
    is not installed', because this step didn't exist yet."""
    if _run(["bash", "-lc", "command -v ffmpeg"], timeout=15).returncode == 0:
        return "already present"
    r = _run(
        ["bash", "-lc", "apt-get update -qq && apt-get install -y -qq ffmpeg"],
        timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(f"apt-get install ffmpeg failed: {r.stderr[-400:]}")
    return "installed via apt-get"


def ensure_facefusion():
    """Skipped when a pre-baked Network Volume already has this venv."""
    ffmpeg_note = ensure_ffmpeg()
    if VENV_PY.is_file():
        return {"bootstrapped": False, "baked_image": FF_DIR == BAKED_FF_DIR, "ff_dir": str(FF_DIR), "ffmpeg": ffmpeg_note}
    steps = [f"ffmpeg: {ffmpeg_note}"]
    r = _run(
        [
            "git",
            "clone",
            "--branch",
            "3.6.0",
            "--depth",
            "1",
            "https://github.com/facefusion/facefusion.git",
            str(FF_DIR),
        ],
        timeout=180,
    )
    steps.append(f"clone rc={r.returncode}")
    if r.returncode != 0:
        raise RuntimeError(f"git clone failed: {r.stderr[-400:]}")
    r = _run(["python3", "-m", "venv", "venv"], cwd=str(FF_DIR), timeout=120)
    steps.append(f"venv rc={r.returncode}")
    pip = str(FF_DIR / "venv" / "bin" / "pip")
    r = _run([pip, "install", "-q", "--upgrade", "pip"], timeout=120)
    # Mirror facefusion/installer.py exactly: every requirement EXCEPT the CPU
    # onnxruntime, then uninstall both runtimes, then the pinned GPU build.
    # Installing requirements.txt as-is put onnxruntime (CPU) and
    # onnxruntime-gpu side by side in one venv, and CUDAExecutionProvider
    # then fails to load — measured on a real pod 2026-09-05 (rc=4, $0.02).
    reqs = [
        ln.strip()
        for ln in open(FF_DIR / "requirements.txt", encoding="utf-8")
        if ln.strip() and not ln.strip().startswith("onnxruntime")
    ]
    r = _run([pip, "install", "-q", *reqs], cwd=str(FF_DIR), timeout=900)
    steps.append(f"requirements rc={r.returncode}")
    if r.returncode != 0:
        raise RuntimeError(f"pip install requirements failed: {r.stderr[-400:]}")
    _run([pip, "uninstall", "-y", "-q", "onnxruntime", "onnxruntime-gpu"], timeout=120)
    ort_pin = "onnxruntime-gpu==1.24.3"  # facefusion/installer.py ONNXRUNTIME_SET['cuda'] for 3.6.0
    r = _run([pip, "install", "-q", ort_pin], timeout=300)
    steps.append(f"{ort_pin} rc={r.returncode} {(r.stderr or '').strip()[-200:]}")
    if r.returncode != 0:
        # 1.24.3 is not on PyPI for linux/py3.11 (pip lists 1.24.1, 1.24.4, ...).
        # Never fall to "latest": 1.29 is built for CUDA 13 (wants
        # libcublasLt.so.13) and this image is CUDA 12.4 — measured
        # 2026-09-05, $0.012. Stay on the CUDA 12 builds nearest the pin.
        for alt in ("onnxruntime-gpu==1.24.4", "onnxruntime-gpu==1.24.1", "onnxruntime-gpu==1.23.2"):
            r = _run([pip, "install", "-q", alt], timeout=300)
            steps.append(f"{alt} rc={r.returncode} {(r.stderr or '').strip()[-160:]}")
            if r.returncode == 0:
                break
        if r.returncode != 0:
            raise RuntimeError(f"pip could not install a CUDA-12 onnxruntime-gpu: {(r.stderr or '')[-400:]}")
    v = _run([str(VENV_PY), "-c", "import onnxruntime as o,sys;print(o.__version__, sys.version.split()[0])"], timeout=60)
    steps.append(f"onnxruntime import: {(v.stdout or v.stderr or '').strip()[-120:]}")
    # ORT 1.24 on CUDA 12 needs cuDNN 9 + cuBLAS/cuFFT/cuRAND + the CUDA
    # runtime beside it; the venv sees none of the image's. The pip wheels
    # put them under site-packages/nvidia/*/lib and ort.preload_dlls() loads
    # them from there.
    r = _run([pip, "install", "-q", "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12"], timeout=600)
    steps.append(f"cuda wheels rc={r.returncode}")
    return {"bootstrapped": True, "steps": steps}


def cuda_provider_ok():
    """True only if onnxruntime can really construct a CUDA session on this pod.
    `get_available_providers()` lists CUDA even when its libraries are missing;
    building a session is the honest test."""
    probe = (
        "import onnxruntime as ort, numpy as np;"
        "getattr(ort, 'preload_dlls', lambda *a, **k: None)();"
        "from onnx import helper, TensorProto;"
        "x=helper.make_tensor_value_info('x',TensorProto.FLOAT,[1]);"
        "g=helper.make_graph([helper.make_node('Identity',['x'],['y'])],'g',[x],[helper.make_tensor_value_info('y',TensorProto.FLOAT,[1])]);"
        "m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)]);"
        "s=ort.InferenceSession(m.SerializeToString(),providers=['CUDAExecutionProvider']);"
        "print(s.get_providers())"
    )
    r = _run([str(VENV_PY), "-c", probe], timeout=120)
    text = (r.stdout or "") + (r.stderr or "")
    ok = "CUDAExecutionProvider" in (r.stdout or "")
    # surface the line that names the missing library, e.g.
    # "Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open..."
    why = next((ln.strip() for ln in text.splitlines() if "error:" in ln.lower() or "cannot open" in ln.lower()), "")
    return ok, (why or text)[-600:]


def build_cmd(args):
    processors = ["face_swapper"]
    if args.expression_factor > 0:
        processors.append("expression_restorer")
    if (
        args.enhancer_model
        and args.enhancer_model.lower() != "none"
        and args.enhancer_blend not in (0,)
    ):
        processors.append("face_enhancer")
    cmd = [
        str(VENV_PY),
        "facefusion.py",
        "headless-run",
        "-s",
        *args.source,
        "-t",
        args.target,
        "-o",
        args.out,
        "--processors",
        *processors,
        "--face-swapper-model",
        args.swapper_model,
        "--face-swapper-pixel-boost",
        args.pixel_boost,
        "--face-selector-mode",
        args.selector_mode,
        "--face-mask-types",
        "box",
        "occlusion",
        "region",
        "--face-occluder-model",
        "xseg_2",
        "--execution-providers",
        "cuda",
    ]
    if "expression_restorer" in processors:
        cmd += [
            "--expression-restorer-model",
            "live_portrait",
            "--expression-restorer-factor",
            str(args.expression_factor),
        ]
    if "face_enhancer" in processors:
        cmd += [
            "--face-enhancer-model",
            args.enhancer_model,
            "--face-enhancer-blend",
            str(args.enhancer_blend),
        ]
    if args.swapper_weight is not None:  # FaceFusion 3.x: 0.0 = lightest swap, 1.0 = full
        cmd += ["--face-swapper-weight", str(args.swapper_weight)]
    if args.selector_mode == "reference" and args.reference_position is not None:
        cmd += [
            "--face-selector-order", "large-small",
            "--reference-face-position", str(args.reference_position),
            "--reference-frame-number", str(args.reference_frame or 0),
            "--reference-face-distance", "0.60",
        ]
    if Path(args.target).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        # libx264, not h264_nvenc: the pod's apt-get ffmpeg has no NVENC, so
        # the merge step failed after a full GPU swap (measured 2026-09-05:
        # 78 frames swapped in 77 s, then "merging video failed", $0.05).
        # Encoding 78 frames on CPU is seconds; quality 90 keeps the swap sharp.
        cmd += ["--output-video-encoder", "libx264", "--output-video-quality", "90"]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, nargs="+")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--swapper-model", default="hyperswap_1a_256")
    ap.add_argument("--pixel-boost", default="512x512")
    ap.add_argument("--enhancer-model", default="codeformer")
    ap.add_argument("--enhancer-blend", type=int, default=50)
    ap.add_argument("--expression-factor", type=int, default=80)
    ap.add_argument("--selector-mode", default="one")
    ap.add_argument("--reference-position", type=int, default=None)
    ap.add_argument("--reference-frame", type=int, default=None)
    ap.add_argument("--swapper-weight", type=float, default=None)
    args = ap.parse_args()

    steps = []
    started = time.time()
    try:
        boot = ensure_facefusion()
        steps.append(f"bootstrap={boot}")
    except Exception as e:
        _emit({"ok": False, "error": f"bootstrap failed: {e}", "steps": steps}, 2)

    # nvidia pip wheels live under site-packages/nvidia/*/lib; put them on the
    # loader path for BOTH the probe and the swap (set before either starts —
    # glibc reads LD_LIBRARY_PATH at process start, so setting it inside a
    # running interpreter does nothing).
    import glob as _glob
    try:
        _sp = subprocess.run([str(VENV_PY), "-c", "import site;print(site.getsitepackages()[0])"], capture_output=True, text=True, timeout=30).stdout.strip()
        libs = ":".join(_glob.glob(_sp + "/nvidia/*/lib"))
        if libs:
            os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + libs
    except Exception:
        pass
    ok_cuda, cuda_note = cuda_provider_ok()
    steps.append(f"cuda provider: {'ok' if ok_cuda else 'MISSING'} {cuda_note.strip()[-600:]}")
    if not ok_cuda:
        # Refuse rather than burn 20 minutes on CPU and get killed with no diagnosis.
        _emit({"ok": False, "error": "onnxruntime cannot use the pod's GPU (CUDAExecutionProvider failed to load) — refusing to run on CPU", "steps": steps}, 4)
        return
    cmd = build_cmd(args)
    steps.append("running headless-run")
    try:
        # Real evidence (2026-08-23, run_id=1c30e94a...): a 1-second/30-frame
        # video clip alone ran past this ceiling and got killed at 1100s with
        # no partial-progress signal captured. Bumped with real headroom
        # instead of guessing at a small nudge, since the timeout kill itself
        # gives no clue how close it was to finishing.
        is_video = Path(args.target).suffix.lower() in (
            ".mp4",
            ".mov",
            ".mkv",
            ".avi",
            ".webm",
        )
        # Stream FaceFusion's stderr so the dispatcher log shows real progress
        # (percent lines) instead of a silent 20-minute wall.
        limit = 2000 if is_video else 1100
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(FF_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        tail = ""
        last_pct = -1
        buf = ""
        while True:
            chunk = proc.stderr.read(256) if proc.stderr else ""
            if not chunk:
                if proc.poll() is not None:
                    break
                if time.time() - t0 > limit:
                    proc.kill()
                    raise subprocess.TimeoutExpired(cmd, limit)
                continue
            buf += chunk
            tail = (tail + chunk)[-4000:]
            for seg in buf.replace("\n", "\r").split("\r")[:-1]:
                m = re.search(r"(\d{1,3})%", seg)
                if m and int(m.group(1)) != last_pct:
                    last_pct = int(m.group(1))
                    print(json.dumps({"progress": last_pct, "elapsed": int(time.time() - t0)}), flush=True)
            buf = buf.split("\r")[-1] if "\r" in buf else buf[-256:]
            if time.time() - t0 > limit:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd, limit)
        rc = proc.wait()

        class _R:  # keep the shape the code below expects
            returncode = rc
            stderr = tail

        r = _R()
    except subprocess.TimeoutExpired:
        _emit(
            {"ok": False, "error": "facefusion headless-run timed out", "steps": steps},
            3,
        )
        return
    ok = (
        r.returncode == 0
        and os.path.isfile(args.out)
        and os.path.getsize(args.out) > 10240
    )
    _emit(
        {
            "ok": ok,
            "error": ""
            if ok
            else f"facefusion exited {r.returncode}: {(r.stderr or '')[-500:]}",
            "seconds": int(time.time() - started),
            "steps": steps,
            "bytes_out": os.path.getsize(args.out) if os.path.isfile(args.out) else 0,
        },
        0 if ok else 1,
    )


if __name__ == "__main__":
    main()
