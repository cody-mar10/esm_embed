#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path

import torch
from esm import pretrained

MODELS = {
    "esm2_t48_15B": pretrained.esm2_t48_15B_UR50D,
    "esm2_t36_3B": pretrained.esm2_t36_3B_UR50D,
    "esm2_t33_650M": pretrained.esm2_t33_650M_UR50D,
    "esm2_t30_150M": pretrained.esm2_t30_150M_UR50D,
    "esm2_t12_35M": pretrained.esm2_t12_35M_UR50D,
    "esm2_t6_8M": pretrained.esm2_t6_8M_UR50D,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ESM2 model(s) if not present"
    )

    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        metavar="MODELS",
        choices=MODELS.keys(),
        default=list(reversed(MODELS.keys())),
        help="ESM2 to download or check (default: all) [choices: %(choices)s]",
    )
    parser.add_argument(
        "-c",
        "--torch-cache",
        type=Path,
        default=Path.cwd().joinpath("models/hub"),
        help="torch cache for storing models (default: %(default)s)",
    )
    return parser.parse_args()


def is_downloaded(model: str, cache: Path) -> bool:
    return cache.joinpath("checkpoints").joinpath(f"{model}_UR50D.pt").exists()


def main(models: list[str], cache: Path):
    torch.hub.set_dir(cache.as_posix())
    for model in models:
        now = datetime.now().strftime("%H:%M:%S")
        if is_downloaded(model, cache):
            print(f"[{now}]: {model} already downloaded", flush=True)
        else:
            print(f"[{now}]: Downloading {model}", flush=True)
            MODELS[model]()
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}]: Finished downloading {model}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(args.models, args.torch_cache)
