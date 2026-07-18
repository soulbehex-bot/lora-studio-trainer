# Wan 2.2 14B LoRA trainer - RunPod serverless worker.
# Uses ai-toolkit (ostris), which downloads Wan 2.2 A14B itself and uint4-
# quantizes it to fit a 24GB GPU. The handler takes a dataset (base64 images or
# URLs) + a trigger word, trains a LoRA, and returns the two expert .safetensors.
#
# IMPORTANT (endpoint config): give this endpoint >=60GB container disk. The
# A14B model is ~28.6GB and ai-toolkit needs room to download + quantize it; a
# 30GB default disk fails with "Disk quota exceeded". See worker README.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# curl is used to push the trained files to 0x0.st when no presigned URL is given
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ai-toolkit + its submodules (this repo carries config/examples/*.yaml we patch)
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git submodule update --init --recursive

# python deps: ai-toolkit's requirements MINUS torch/vision/audio (the base image
# already ships them; letting ai-toolkit reinstall torch roughly doubled the
# image size and made cold starts unusable). Then the runpod SDK + pyyaml/hf.
# Finally UNINSTALL hf_xet: its "Reconstructing" download path needs ~2x the
# file size on disk (and ignores HF_HUB_DISABLE_XET in some versions), which is
# what made the 28.6GB model download blow past the disk on the first attempts.
RUN cd ai-toolkit && \
    sed -i -E '/^(torch|torchvision|torchaudio)([<>=~! ]|$)/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod huggingface_hub pyyaml && \
    pip uninstall -y hf_xet || true && \
    pip cache purge || true

# The Wan 2.2 A14B model (ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16) and the
# uint4 accuracy-recovery adapter download on the first job into HF_HOME
# (/app/hf, on the container disk) and are reused by warm workers.

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
