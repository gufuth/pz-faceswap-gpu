"""Pre-download the FaceFusion 3.6.0 models the Project Zero Player can ask for.

Runs inside /opt/facefusion with its venv at image build time. Uses FaceFusion's
own model tables and downloader (the same code its headless-run pre_check calls),
so the files land in .assets/models exactly where a run looks for them and the
.hash files match. Scope: every common module (detector, landmarker, masker incl.
xseg_2 + region parser, recognizer, classifier, content analyser, voice extractor)
plus the swappers / enhancers the Player exposes and live_portrait.
"""
import os
import sys

sys.path.insert(0, os.getcwd())  # run from /opt/facefusion; the script itself lives in /opt/pz

import facefusion.core as core
from facefusion import content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, voice_extractor
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.processors.core import get_processors_modules

# pz-app/pz_faceswap.py SWAPPER_MODELS / ENHANCER_MODELS; None = the whole set.
WANT = {
    "face_swapper": ["hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "ghost_3_256", "inswapper_128"],
    "face_enhancer": ["codeformer", "restoreformer_plus_plus", "gpen_bfr_2048", "gfpgan_1.4"],
    "expression_restorer": None,
}


def fetch(model, label):
    hs, ss = model.get("hashes"), model.get("sources")
    if not (hs and ss):
        return True
    ok = conditional_download_hashes(hs) and conditional_download_sources(ss)
    print(("OK   " if ok else "FAIL ") + label, flush=True)
    return ok


def bake():
    ok = True
    for mod in (content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, voice_extractor):
        for name, model in mod.create_static_model_set("full").items():
            ok = fetch(model, f"{mod.__name__}:{name}") and ok
    for proc, names in WANT.items():
        mod = get_processors_modules([proc])[0]
        table = mod.create_static_model_set("full")
        for name in names or list(table):
            if name not in table:
                print(f"FAIL {proc}:{name} not in FaceFusion 3.6.0 model table", flush=True)
                ok = False
                continue
            ok = fetch(table[name], f"{proc}:{name}") and ok
    return 0 if ok else 1


core.force_download = bake  # route() calls it after FaceFusion's own arg/state setup
sys.argv = ["facefusion.py", "force-download", "--download-scope", "full", "--log-level", "info"]
core.cli()
