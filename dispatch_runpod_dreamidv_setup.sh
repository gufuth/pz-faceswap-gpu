#!/bin/bash
# dispatch_runpod_dreamidv_setup.sh — runs ON the RunPod pod, launched detached by
# dispatch_runpod_dreamidv.py (setsid nohup ... </dev/null). Writes /workspace/STATUS,
# which the host polls. Idempotent: every install step drops a marker, so a re-launch
# resumes instead of starting over.
#
# Recipe proven by two measured pod runs on 2026-09-26 (bake-off, L40S 48GB,
# image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04, 80 GB disk):
#   clone bytedance/DreamID-V; install requirements minus torch/torchvision/flash_attn/
#   xfuser/mediapipe; patch the Wan attention to torch SDPA (no flash_attn wheel for this
#   image); fetch dreamidv_faster.pth + the two DWPose onnx files + Wan2.1_VAE.pth only
#   (the Faster script ships context.pth, so no T5 encoder is needed).
#
# Inputs (uploaded by the host):
#   /workspace/in/ref.<ext>          one reference face (a 512x512 crop is recommended)
#   /workspace/in/wNNN.mp4           1280x720 windows, each 4n+1 frames, <= 81 frames
#   /workspace/in/windows.txt        one line per window: "<NNN> <frame_count>"
#   /workspace/in/params.txt         "STEPS=16 SEED=42 FPS=25 REF=/workspace/in/ref.png"
# Outputs:
#   /workspace/out/wNNN.mp4          the swapped window, same frame count as its input
set -o pipefail
W=/workspace
st() { echo "$1" > $W/STATUS; echo "=== STATUS $1 $(date +%T)"; }
fail() { st "FAIL:$1"; exit 1; }
cd $W

# Pre-baked image (ghcr.io/gufuth/pz-faceswap-gpu:latest): the install steps below
# were already run at image build time (this same script, PZ_SETUP_ONLY=1) and the
# results moved to /opt/dreamidv. Link them into /workspace and mark every install
# step done, so a run goes straight to the render. The bare pytorch image has no
# /opt/dreamidv/.baked and falls through to the full install.
B=/opt/dreamidv
if [ -f $B/.baked ] && [ ! -f $W/.m_dl ]; then
  st setup_baked
  for d in DreamID-V models wan; do
    [ -e $W/$d ] || ln -s $B/$d $W/$d
  done
  touch $W/.m_clone $W/.m_pip $W/.m_patch $W/.m_dl
  echo "BAKED_IMAGE $(cat $B/.baked)"
fi

if [ ! -f $W/.m_clone ]; then
  st setup_clone
  rm -rf $W/DreamID-V
  git clone --depth 1 https://github.com/bytedance/DreamID-V $W/DreamID-V || fail setup_clone
  touch $W/.m_clone
fi
cd $W/DreamID-V

if [ ! -f $W/.m_pip ]; then
  st setup_pip
  # keep the image's torch 2.4; skip flash_attn (patched to SDPA below), xfuser (multi-GPU
  # only), mediapipe (the Faster script uses DWPose; its pin line carries an inline comment)
  grep -v -E '^(torch|torchvision|flash_attn|xfuser|mediapipe)' requirements.txt > $W/req2.txt
  pip install -q -r $W/req2.txt decord einops matplotlib scikit-image scipy huggingface_hub hf_transfer 2>&1 | tail -20 || fail setup_pip
  python -c "import decord, diffusers, transformers, cv2, onnxruntime, easydict, imageio; print('imports ok')" || fail setup_pip_import
  for i in 1 2 3 4 5 6; do
    out=$(python generate_dreamidv_faster.py --help 2>&1)
    m=$(echo "$out" | grep -oP "No module named '\K[^'.]+" | head -1)
    [ -z "$m" ] && { echo "IMPORTS_OK iter $i"; break; }
    case $m in skimage) p=scikit-image;; yaml) p=pyyaml;; PIL) p=pillow;; cv2) p=opencv-python;; sklearn) p=scikit-learn;; *) p=$m;; esac
    echo "auto-installing $p"; pip install -q $p 2>&1 | tail -2
  done
  # Image build (PZ_SETUP_ONLY=1) runs on a GPU-less host: the script imports cleanly
  # (no missing module above) and only stops at CUDA init, which a real pod passes.
  if ! echo "$out" | grep -q "usage:"; then
    if [ "${PZ_SETUP_ONLY:-0}" = "1" ] && echo "$out" | grep -q "Found no NVIDIA driver"; then
      echo "IMPORT_CHECK build host has no GPU: imports ok up to CUDA init"
    else
      echo "$out" | tail -30; fail setup_import_check
    fi
  fi
  touch $W/.m_pip
fi

if [ ! -f $W/.m_patch ]; then
  st setup_patch
  # Wan flash_attention() -> torch SDPA fallback, per sample, honouring q_lens/k_lens.
  # Same code as the bake-off's sdpa_patch.py (verified there against a plain implementation).
  python - dreamidv_wan_faster/modules/attention.py <<'PYEOF' || fail setup_patch
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "        assert FLASH_ATTN_2_AVAILABLE\n"
if "PZ_SDPA_FALLBACK" in s:
    print("already patched"); sys.exit(0)
assert anchor in s, "anchor not found"
fallback = """        if not FLASH_ATTN_2_AVAILABLE:  # PZ_SDPA_FALLBACK
            outs, qs, ks = [], 0, 0
            for i in range(b):
                ql, kl = int(q_lens[i]), int(k_lens[i])
                qi = q[qs:qs + ql].transpose(0, 1).unsqueeze(0)
                ki = k[ks:ks + kl].transpose(0, 1).unsqueeze(0)
                vi = v[ks:ks + kl].transpose(0, 1).unsqueeze(0)
                oi = torch.nn.functional.scaled_dot_product_attention(
                    qi, ki, vi, is_causal=causal, scale=softmax_scale)
                oi = oi.squeeze(0).transpose(0, 1)
                if ql < lq:
                    oi = torch.cat([oi, oi.new_zeros((lq - ql,) + tuple(oi.shape[1:]))])
                outs.append(oi)
                qs += ql
                ks += kl
            return torch.stack(outs).type(out_dtype)
"""
s = s.replace(anchor, fallback + anchor, 1)
open(p, "w").write(s)
print("patched")
PYEOF
  touch $W/.m_patch
fi

if [ ! -f $W/.m_dl ]; then
  st setup_download
  export HF_HUB_ENABLE_HF_TRANSFER=1
  python - <<'PYEOF' || fail setup_download
from huggingface_hub import hf_hub_download
import shutil, os, time
t = time.time()
hf_hub_download("XuGuo699/DreamID-V", "dreamidv_faster.pth", local_dir="/workspace/models")
for f in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
    q = hf_hub_download("XuGuo699/DreamID-V", f, local_dir="/workspace/models")
    os.makedirs("/workspace/DreamID-V/pose/models", exist_ok=True)
    shutil.copy(q, "/workspace/DreamID-V/pose/models/" + f)
hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth", local_dir="/workspace/wan")
for d in ("/workspace/models", "/workspace/wan", "/workspace/DreamID-V/pose/models"):
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isfile(fp):
            print(fp, os.path.getsize(fp))
print("download s", round(time.time() - t))
PYEOF
  touch $W/.m_dl
fi

# Image build: install only, no render.
if [ "${PZ_SETUP_ONLY:-0}" = "1" ]; then st INSTALLED; exit 0; fi

# ---- per-window render --------------------------------------------------------------
. $W/in/params.txt
[ -s "$REF" ] || fail no_ref
[ -s $W/in/windows.txt ] || fail no_windows
mkdir -p $W/out
N=$(grep -c . $W/in/windows.txt)
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
while read -r IDX LEN; do
  [ -z "$IDX" ] && continue
  OUT=$W/out/w$IDX.mp4
  if [ -s "$OUT" ]; then echo "skip w$IDX (exists)"; continue; fi
  st "run:$IDX:$N"
  T0=$(date +%s)
  python generate_dreamidv_faster.py --size 1280*720 --ckpt_dir $W/wan \
    --dreamidv_ckpt $W/models/dreamidv_faster.pth --sample_steps $STEPS --base_seed $SEED \
    --frame_num $LEN --ref_image "$REF" --ref_video $W/in/w$IDX.mp4 --sample_fps $FPS \
    --offload_model False --save_file $OUT < /dev/null || fail "run:$IDX"
  echo "WINDOW_SECONDS $IDX $(( $(date +%s) - T0 ))"
  [ -s "$OUT" ] || fail "no_output:$IDX"
done < $W/in/windows.txt
st DONE
