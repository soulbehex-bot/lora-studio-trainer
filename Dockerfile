# Wan 2.2 14B LoRA trainer - RunPod serverless worker.
# Uses ai-toolkit (ostris), which downloads the Wan 2.2 14B model itself and
# quantizes it to fit a 24GB GPU. The handler takes a dataset (image URLs) + a
# trigger word, trains a LoRA, and uploads the .safetensors.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# ai-toolkit + its submodules
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git submodule update --init --recursive

# python deps: ai-toolkit's own requirements + the runpod SDK + pyyaml/hf
RUN cd ai-toolkit && pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod huggingface_hub pyyaml

# The Wan 2.2 14B model (ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16) and the uint4
# accuracy-recovery adapter download on the first job into the HF cache and are
# reused by warm workers. (Baking them in would make the image ~30GB+; the
# network volume / warm-worker cache keeps cold starts to the first run only.)

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
