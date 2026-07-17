"""
Wan 2.2 14B LoRA trainer - RunPod serverless handler.

Input (event["input"]):
  dataset_urls : [str]   image URLs to train on (required)
  trigger_word : str     token written into every caption (default "mylora")
  name         : str     LoRA name / output filename (default random)
  steps        : int     training steps (default 1000; keep low for test runs)
  result_upload: str     "0x0" (default, temporary public host for tests) or
                         an R2/S3 presigned PUT url the app provides for real use

Output:
  { lora_url, filename, size, trigger, steps }  on success
  { error, stderr, stdout }                     on failure

The config mirrors ai-toolkit's official train_lora_wan22_14b_24gb.yaml: 14B
model auto-downloaded + uint4-quantized to fit 24GB, both noise experts trained.
"""

import glob
import os
import subprocess
import urllib.request
import uuid

import runpod
import yaml

AITK = "/app/ai-toolkit"
OUT = "/app/output"


def download_dataset(urls, dest):
    os.makedirs(dest, exist_ok=True)
    paths = []
    for i, u in enumerate(urls):
        ext = os.path.splitext(u.split("?")[0])[1].lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        p = os.path.join(dest, "img_%03d%s" % (i, ext))
        urllib.request.urlretrieve(u, p)
        paths.append(p)
    return paths


def write_captions(image_paths, trigger):
    for p in image_paths:
        base = os.path.splitext(p)[0]
        with open(base + ".txt", "w") as f:
            f.write(trigger)


def build_config(name, dataset_dir, steps):
    return {
        "job": "extension",
        "config": {
            "name": name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": OUT,
                "device": "cuda:0",
                "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
                "save": {"dtype": "float16", "save_every": steps + 1, "max_step_saves_to_keep": 1},
                "datasets": [{
                    "folder_path": dataset_dir,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "num_frames": 1,
                    "resolution": [512, 768, 1024],
                }],
                "train": {
                    "batch_size": 1,
                    "steps": steps,
                    "gradient_accumulation": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "timestep_type": "linear",
                    "optimizer": "adamw8bit",
                    "lr": 1e-4,
                    "optimizer_params": {"weight_decay": 1e-4},
                    "switch_boundary_every": 10,
                    "cache_text_embeddings": True,
                    "dtype": "bf16",
                },
                "model": {
                    "name_or_path": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
                    "arch": "wan22_14b",
                    "quantize": True,
                    "qtype": "uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors",
                    "quantize_te": True,
                    "qtype_te": "qfloat8",
                    "low_vram": True,
                    "model_kwargs": {"train_high_noise": True, "train_low_noise": True},
                },
                # no sampling during a headless training run (saves time/VRAM)
                "sample": {"sample_every": steps + 1, "prompts": [], "sample_steps": 25},
            }],
        },
        "meta": {"name": name, "version": "1.0"},
    }


def upload_result(path, mode):
    # For test runs, push to 0x0.st (temporary public host) and return the URL.
    # For real use the app passes an R2/S3 presigned PUT url instead.
    if mode and mode.startswith("http"):
        with open(path, "rb") as f:
            req = urllib.request.Request(mode, data=f.read(), method="PUT")
            urllib.request.urlopen(req)
        return mode.split("?")[0]
    r = subprocess.run(["curl", "-fsS", "-F", "file=@%s" % path, "https://0x0.st"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def handler(event):
    inp = event.get("input", {}) or {}
    urls = inp.get("dataset_urls") or []
    if not urls:
        return {"error": "no dataset_urls provided"}
    trigger = inp.get("trigger_word") or "mylora"
    name = inp.get("name") or ("lora_" + uuid.uuid4().hex[:8])
    steps = int(inp.get("steps") or 1000)

    ds = os.path.join("/app/datasets", name)
    try:
        imgs = download_dataset(urls, ds)
    except Exception as e:
        return {"error": "dataset download failed: %s" % e}
    if not imgs:
        return {"error": "no images downloaded"}
    write_captions(imgs, trigger)

    cfg = build_config(name, ds, steps)
    cfg_path = os.path.join("/app", name + ".yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    proc = subprocess.run(["python", "run.py", cfg_path], cwd=AITK,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": "training failed",
                "stderr": (proc.stderr or "")[-3500:],
                "stdout": (proc.stdout or "")[-1500:]}

    found = glob.glob(os.path.join(OUT, name, "**", "*.safetensors"), recursive=True) \
        or glob.glob(os.path.join(OUT, "**", "*.safetensors"), recursive=True)
    if not found:
        return {"error": "no safetensors produced", "stdout": (proc.stdout or "")[-2500:]}
    lora = sorted(found)[-1]

    try:
        url = upload_result(lora, inp.get("result_upload"))
    except Exception as e:
        return {"error": "trained ok but upload failed: %s" % e,
                "filename": os.path.basename(lora), "size": os.path.getsize(lora)}

    return {"lora_url": url, "filename": os.path.basename(lora),
            "size": os.path.getsize(lora), "trigger": trigger, "steps": steps}


runpod.serverless.start({"handler": handler})
