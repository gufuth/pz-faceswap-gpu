# pz-faceswap-gpu

Pre-installed RunPod image for the Project Zero Player's face-swap engines, so a run skips
its from-scratch install.

- `ghcr.io/gufuth/pz-faceswap-gpu:facefusion`: Classic. FaceFusion 3.6.0 at `/opt/facefusion`
  (venv, onnxruntime-gpu, and models: hyperswap 1a/1b/1c, ghost_3, inswapper_128, codeformer,
  restoreformer++, gpen_bfr_2048, gfpgan_1.4, live_portrait, xseg, parsers, detectors).
- `ghcr.io/gufuth/pz-faceswap-gpu:latest`: the above plus DreamID-V (Faster) at `/opt/dreamidv`.

`pz_faceswap_runpod_runner.py` and `dispatch_runpod_dreamidv_setup.sh` are copies of the
PROJECT ZERO `.claude/commands/` files of the same name. The build runs their own install
code, so keep them in sync. A push to `main` rebuilds via GitHub Actions.
