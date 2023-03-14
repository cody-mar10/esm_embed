from __future__ import annotations

import os
import re
import subprocess

GPU_DEVICE_NO = re.compile(r"GPU (\d):")
GPU_UUID = re.compile(r"UUID: (GPU-\w+)-")


def get_visible_devices() -> list[str]:
    try:
        return [
            gpu.lstrip().rstrip()
            for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        ]
    except KeyError as e:
        print(
            "CUDA_VISIBLE_DEVICES not found as an env var, suggesting that no GPUs are available."
        )
        raise e


def map_gpu_ids() -> dict[str, int]:
    visible_devices = get_visible_devices()

    gpu_info: dict[str, int] = dict()
    all_gpus = subprocess.check_output(["nvidia-smi", "-L"]).decode().splitlines()

    for gpu in all_gpus:
        device_id = int(GPU_DEVICE_NO.findall(gpu)[0])
        uuid = GPU_UUID.findall(gpu)[0]
        gpu_info[uuid] = device_id

    return gpu_info
