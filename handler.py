"""
Wan 2.2 14B LoRA trainer - RunPod Serverless handler.

This mirrors the pipeline proven end-to-end on a RunPod RTX 4090 pod on
2026-07-18: ai-toolkit (ostris) downloads Wan 2.2 A14B, uint4-quantizes it to
fit a 24GB GPU, and trains a LoRA. A 14B LoRA is TWO files (the MoE's two noise
experts: high_noise + low_noise) - both are returned.

Input (event["input"]):
  dataset_b64  : [{name, data}]  base64 images (preferred for small sets - no
                                  external host needed). data may be a bare
                                  base64 string or a data: URI.
  dataset_urls : [str]           image URLs (alternative to dataset_b64)
  trigger_word : str             token written into every caption (default "mylora")
  name         : str             LoRA name / output stem (default random)
  steps        : int             training steps (default 500; 500-4000 is the useful range)
  result_upload_high / result_upload_low : str
                                 optional presigned S3/R2 PUT urls for the two
                                 expert files (durable hosting). If omitted, both
                                 are pushed to 0x0.st (temporary public host,
                                 fine for a client that downloads immediately).

Output on success:
  { high_url, low_url, high_name, low_name, size, trigger, steps }
Output on failure:
  { error, stderr, stdout }

Why this exact recipe: it patches ai-toolkit's OFFICIAL example
`config/examples/train_lora_wan22_14b_24gb.yaml` (uint4 + accuracy-recovery
adapter + low_vram, both experts trained) instead of hand-building a config -
that official recipe is the one that trained cleanly on the 4090. We only patch
the dataset, name, steps, and disable sampling (see the traps below).
"""

import glob
import os

# --- disk traps learned the hard way (see reference_wan22_lora_training) ------
# 1. Keep the HuggingFace cache on the big container disk, never a small
#    network-volume default. On serverless the container disk (endpoint-
#    configured, >=60GB) lives under /app.
os.environ.setdefault("HF_HOME", "/app/hf")
# 2. xet's "Reconstructing" step needs ~2x the file size on disk and ignores
#    HF_HUB_DISABLE_XET in some versions; the Dockerfile uninstalls hf_xet so
#    downloads are a clean 1x. This flag is belt-and-suspenders.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import subprocess
import urllib.request
import uuid
import base64

import runpod
import yaml

AITK = "/app/ai-toolkit"
OUT = "/app/output"
EXAMPLE_CONFIG = os.path.join(
    AITK, "config", "examples", "train_lora_wan22_14b_24gb.yaml"
)


def _write_image(dest_dir, index, raw_bytes, ext=".jpg"):
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    p = os.path.join(dest_dir, "img_%03d%s" % (index, ext))
    with open(p, "wb") as f:
        f.write(raw_bytes)
    return p


def stage_dataset(inp, dest):
    """Materialize the dataset into `dest` from base64 items or URLs."""
    os.makedirs(dest, exist_ok=True)
    paths = []
    b64_items = inp.get("dataset_b64") or []
    for i, item in enumerate(b64_items):
        data = item.get("data") if isinstance(item, dict) else item
        name = (item.get("name") if isinstance(item, dict) else None) or ("img_%03d.jpg" % i)
        if isinstance(data, str) and data.startswith("data:"):
            data = data[data.find(",") + 1:]
        raw = base64.b64decode(data)
        ext = os.path.splitext(name)[1] or ".jpg"
        paths.append(_write_image(dest, i, raw, ext))
    urls = inp.get("dataset_urls") or []
    base = len(paths)
    for j, u in enumerate(urls):
        ext = os.path.splitext(u.split("?")[0])[1].lower() or ".jpg"
        raw = urllib.request.urlopen(u, timeout=120).read()
        paths.append(_write_image(dest, base + j, raw, ext))
    return paths


def write_captions(image_paths, trigger):
    for p in image_paths:
        base = os.path.splitext(p)[0]
        with open(base + ".txt", "w") as f:
            f.write(trigger)


def build_config(name, dataset_dir, steps):
    """Load ai-toolkit's official 14B example and patch only what we must.

    Patches (each one is a trap we hit on the proven run):
      - datasets[0].folder_path -> our staged dataset
      - name / training_folder  -> our output
      - train.steps             -> requested steps
      - save.save_every         -> steps (guarantees a final checkpoint)
      - sample.prompts = []     -> ai-toolkit renders a baseline sample per
                                   prompt at step 0 (~18 min for the 10 stock
                                   prompts) regardless of sample_start_step;
                                   emptying the list is the only way to skip it.
      - sample.sample_every / sample_start_step pushed past `steps`
    """
    with open(EXAMPLE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg["config"]["name"] = name
    proc = cfg["config"]["process"][0]
    proc["training_folder"] = OUT

    ds = (proc.get("datasets") or [{}])
    ds[0]["folder_path"] = dataset_dir
    proc["datasets"] = ds

    proc.setdefault("train", {})["steps"] = int(steps)

    save = proc.setdefault("save", {})
    save["save_every"] = int(steps)
    save["max_step_saves_to_keep"] = 1

    sample = proc.setdefault("sample", {})
    sample["prompts"] = []
    sample["sample_every"] = int(steps) + 100000
    sample["sample_start_step"] = int(steps) + 100000

    cfg.setdefault("meta", {})["name"] = name
    return cfg


def upload_result(path, presigned):
    """Durable presigned PUT if provided, else 0x0.st (temporary public host)."""
    if presigned and presigned.startswith("http"):
        with open(path, "rb") as f:
            req = urllib.request.Request(presigned, data=f.read(), method="PUT")
            urllib.request.urlopen(req, timeout=600)
        return presigned.split("?")[0]
    r = subprocess.run(
        ["curl", "-fsS", "-F", "file=@%s" % path, "https://0x0.st"],
        capture_output=True, text=True,
    )
    url = (r.stdout or "").strip()
    if not url.startswith("http"):
        raise RuntimeError("0x0 upload failed: %s" % (r.stderr or "")[-200:])
    return url


def _find_expert(out_dir, name, which):
    """Locate the final <name>_<which>_noise.safetensors, preferring the
    un-stepped final file over numbered checkpoints."""
    pats = [
        os.path.join(out_dir, name, "**", "*%s_noise.safetensors" % which),
        os.path.join(out_dir, "**", "*%s_noise.safetensors" % which),
    ]
    hits = []
    for pat in pats:
        hits += glob.glob(pat, recursive=True)
    if not hits:
        return None
    # final file has no step digits in the stem; prefer it, else the highest step
    def rank(p):
        stem = os.path.basename(p)
        has_step = any(ch.isdigit() for ch in stem.replace("_noise", ""))
        return (0 if not has_step else 1, stem)
    hits.sort(key=rank)
    return hits[0]


def handler(event):
    inp = event.get("input", {}) or {}
    if not (inp.get("dataset_b64") or inp.get("dataset_urls")):
        return {"error": "no dataset provided (dataset_b64 or dataset_urls)"}
    trigger = inp.get("trigger_word") or "mylora"
    name = (inp.get("name") or ("lora_" + uuid.uuid4().hex[:8]))
    name = "".join(c for c in name if c.isalnum() or c in "-_") or "lora"
    steps = int(inp.get("steps") or 500)

    ds = os.path.join("/app/datasets", name)
    try:
        imgs = stage_dataset(inp, ds)
    except Exception as e:
        return {"error": "dataset staging failed: %s" % e}
    if not imgs:
        return {"error": "no images in dataset"}
    write_captions(imgs, trigger)

    cfg = build_config(name, ds, steps)
    cfg_path = os.path.join("/app", name + ".yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    proc = subprocess.run(
        ["python", "run.py", cfg_path], cwd=AITK, env=dict(os.environ),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"error": "training failed",
                "stderr": (proc.stderr or "")[-3500:],
                "stdout": (proc.stdout or "")[-1500:]}

    high = _find_expert(OUT, name, "high")
    low = _find_expert(OUT, name, "low")
    if not high or not low:
        return {"error": "expected two expert files, missing one",
                "high": high, "low": low,
                "stdout": (proc.stdout or "")[-2000:]}

    try:
        high_url = upload_result(high, inp.get("result_upload_high"))
        low_url = upload_result(low, inp.get("result_upload_low"))
    except Exception as e:
        return {"error": "trained ok but upload failed: %s" % e,
                "high_name": os.path.basename(high),
                "low_name": os.path.basename(low)}

    return {
        "high_url": high_url,
        "low_url": low_url,
        "high_name": os.path.basename(high),
        "low_name": os.path.basename(low),
        "size": os.path.getsize(high),
        "trigger": trigger,
        "steps": steps,
    }


runpod.serverless.start({"handler": handler})
