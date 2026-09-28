# pz-faceswap-gpu — the Project Zero Player's two RunPod face-swap engines, pre-installed.
#
#   :facefusion  Classic  = FaceFusion 3.6.0 + venv + onnxruntime-gpu + the models the Player uses
#   :latest      Classic + Rebuild = the above + DreamID-V (Faster) + its ~6 GB of weights
#
# Both are built FROM the exact base the dispatchers rented bare before, and the install
# steps are the dispatchers' OWN code run at build time (pz_faceswap_runpod_runner.py's
# ensure_facefusion(), dispatch_runpod_dreamidv_setup.sh with PZ_SETUP_ONLY=1), so a
# baked pod and a from-scratch pod carry the same versions.
#
# Lessons carried over from black-switch-gpu (PZ codex INFR-01/06/13):
#  - bake to a SOURCE dir (/opt), never the runtime dir (/workspace may be a mount);
#  - set NO ENTRYPOINT/CMD: the base image's start.sh does the SSH (PUBLIC_KEY) setup and
#    the dispatchers drive everything over SSH;
#  - never hand-install on a pod.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 AS facefusion

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update -qq && apt-get install -y -qq curl git

COPY pz_faceswap_runpod_runner.py bake_facefusion_models.py /opt/pz/
# The runner's own bootstrap (apt ffmpeg, clone 3.6.0, venv, requirements minus CPU ORT,
# pinned onnxruntime-gpu with its CUDA-12 fallbacks, nvidia cuda wheels), aimed at /opt.
RUN cd /opt/pz && python3 - <<'PY'
import json, sys
from pathlib import Path
import pz_faceswap_runpod_runner as r
r.FF_DIR = Path("/opt/facefusion")
r.VENV_PY = r.FF_DIR / "venv" / "bin" / "python"
res = r.ensure_facefusion()
print(json.dumps(res, indent=1))
steps = " ".join(res.get("steps", []))
if not r.VENV_PY.is_file() or "onnxruntime import:" not in steps:
    sys.exit("facefusion bootstrap incomplete")
PY
RUN /opt/facefusion/venv/bin/python -c "import onnxruntime as o, onnx, cv2, numpy; print('ORT', o.__version__, 'onnx', onnx.__version__, 'numpy', numpy.__version__)" && \
    git -C /opt/facefusion log -1 --format='facefusion %H %d'
RUN cd /opt/facefusion && venv/bin/python /opt/pz/bake_facefusion_models.py && \
    ls -la .assets/models && du -sh .assets/models && \
    echo "facefusion 3.6.0 baked $(date -u +%FT%TZ)" > /opt/facefusion/.baked

FROM facefusion AS full
COPY dispatch_runpod_dreamidv_setup.sh /opt/pz/
# The Rebuild setup script's own install steps (clone, pip, SDPA patch, weights), run into
# /workspace at build time, then moved to /opt/dreamidv where the script links them from.
RUN python3 -c "import torch;print('torch before', torch.__version__)" && \
    mkdir -p /workspace && sed -i 's/\r$//' /opt/pz/dispatch_runpod_dreamidv_setup.sh && \
    PZ_SETUP_ONLY=1 bash /opt/pz/dispatch_runpod_dreamidv_setup.sh && \
    test "$(cat /workspace/STATUS)" = INSTALLED && \
    python3 -c "import torch;print('torch after', torch.__version__)" && \
    mkdir -p /opt/dreamidv && \
    mv /workspace/DreamID-V /workspace/models /workspace/wan /opt/dreamidv/ && \
    mkdir -p /opt/pz/build_markers && mv /workspace/.m_* /workspace/STATUS /workspace/req2.txt /opt/pz/build_markers/ && \
    ls -la /opt/dreamidv/models /opt/dreamidv/wan /opt/dreamidv/DreamID-V/pose/models && \
    echo "dreamid-v $(git -C /opt/dreamidv/DreamID-V rev-parse HEAD) baked $(date -u +%FT%TZ)" > /opt/dreamidv/.baked && \
    cat /opt/dreamidv/.baked
