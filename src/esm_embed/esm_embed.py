#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import esm
import lightning as L
import torch
from lightning.pytorch.accelerators import find_usable_cuda_devices
from torch.utils.data import DataLoader

from . import arch
from .arch.model import MODELS, MODELVALUES
from .utils import _skip_prgm_and_command, chtc_utils


@dataclass
class Args:
    input: Path
    model: MODELVALUES
    batch_size: int
    torch_hub: Path
    outdir: Path
    devices: int
    accelerator: Literal["cpu", "gpu", "auto"]
    precision: Literal[64, 32, 16, "bf16"]

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace):
        return cls(
            input=namespace.input,
            model=namespace.model,
            batch_size=namespace.batch_token_size,
            torch_hub=namespace.torch_hub,
            outdir=namespace.outdir,
            devices=namespace.devices,
            accelerator=namespace.accelerator,
            precision=namespace.precision,
        )


def _add_args(parser: argparse.ArgumentParser):
    required_args = parser.add_argument_group("REQUIRED")
    model_args = parser.add_argument_group("MODEL")

    required_args.add_argument(
        "-i",
        "--input",
        metavar="FILE",
        type=Path,
        required=True,
        help="input protein fasta file",
    )

    model_args.add_argument(
        "-m",
        "--model",
        default="esm2_t6_8M",
        metavar="MODEL",
        choices=MODELS.keys(),
        help="ESM-2 model (default: %(default)s) [%(choices)s]",
    )
    model_args.add_argument(
        "-b",
        "--batch-token-size",
        metavar="INT",
        default=1024,
        type=int,
        help="number of tokens per batch (default: %(default)s)",
    )
    default_torch_hub = Path(
        os.environ.get("TORCH_HOME", "")
        or (
            os.path.join(xdg_dir, "torch/hub")
            if (xdg_dir := os.environ.get("XDG_CACHE_HOME", ""))
            else ""
        )
        or "~/.cache/torch/hub"
    )
    model_args.add_argument(
        "-th",
        "--torch-hub",
        metavar="DIR",
        type=Path,
        default=default_torch_hub,
        help="path to the checkpoints/ directory with downloaded models (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        metavar="DIR",
        type=Path,
        default="out",
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--devices",
        metavar="INT",
        type=int,
        help="number of cpus/gpus to use. If using cpus, this will convert to threads on a single cpu node. If using gpus, this will read the request number of gpus from the CUDA_VISIBLE_DEVICES env var. (default: %(default)s)",
    )
    parser.add_argument(
        "-a",
        "--accelerator",
        metavar="DEVICE",
        default="auto",
        choices={"cpu", "gpu", "auto"},
        help="type of device to use (default: %(default)s) [choices: %(choices)s]",
    )
    parser.add_argument(
        "-p",
        "--precision",
        metavar="PRECISION",
        default=32,
        choices={64, 32, 16, "bf16"},
        type=lambda x: int(x) if x.isdigit() else x,
        help="floating point precision: (deafult: %(default)s) [choices: %(choices)s]",
    )
    return parser


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Embed protein sequences using ESM-2 from Meta AI"
    )
    _add_args(parser)
    args = parser.parse_args(_skip_prgm_and_command())
    return Args(
        input=args.input,
        model=args.model,
        batch_size=args.batch_token_size,
        torch_hub=args.torch_hub,
        outdir=args.outdir,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
    )


# PREDICT MAIN
# TODO: make one for concatenating results
def main():
    args = parse_args()
    torch.hub.set_dir(args.torch_hub)
    L.seed_everything(111)

    model = arch.model.ESM2.from_model_name(args.model)
    data = arch.data.SequenceDataset(
        data=esm.FastaBatchedDataset.from_file(args.input),
        alphabet=model.alphabet,
        batch_size=args.batch_size,
    )
    writer = arch.writer.PredictionWriter(outdir=args.outdir)
    dataloader = DataLoader(
        dataset=data,
        # dataset is already pre-batched
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=arch.data.collate_token_batches,
    )

    if args.accelerator == "cpu":
        torch.set_num_threads(args.devices)
        # half-precision not available on cpus?
        args.precision = 32
        parallelism_kwargs = {"devices": 1}
    elif args.accelerator == "gpu":
        # NOTE: CUDA_VISIBLE_DEVICES shows what GPUs are available
        # in shared systems/clusters, this is only assigned with as many GPUs
        # as requested and shouldn't be edited since that could result in
        # multiple jobs sharing the same GPU.
        # CHTC = CUDA_VISIBLE_DEVICES assigned UUID
        # can map GPU UUID to device no using nvidia-smi -L
        # example output:
        # GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-053b14fc-1ed2-3761-5d42-b3b824feb7d6)
        # GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-1f9fc54e-ce03-79a8-c2ab-4144eb36f0ba)
        # GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-ff708371-0893-0d85-ae2b-0810d095e111)
        # GPU 3: NVIDIA A100-SXM4-80GB (UUID: GPU-7067f38b-b4c2-5568-ec82-f698473ddd07)
        # the assigned GPU IDs would be: "GPU-053b14fc"

        visible_devices = chtc_utils.get_visible_devices()

        # CHTC only makes visible as many devices as requested
        assert (
            len(visible_devices) == args.devices
        ), f"Number of requested GPUs ({args.devices}) does not match the available number of GPUs ({len(visible_devices)})"

        gpu_name_to_gpu_device_no = chtc_utils.map_gpu_ids()
        devices = [gpu_name_to_gpu_device_no[gpu] for gpu in visible_devices]
        parallelism_kwargs = {"devices": devices}
    else:
        parallelism_kwargs = {"devices": "auto"}

    trainer = L.Trainer(
        enable_checkpointing=False,
        callbacks=[writer],
        accelerator=args.accelerator,
        logger=False,
        enable_progress_bar=False,
        precision=args.precision,
        **parallelism_kwargs,
    )

    trainer.predict(model=model, dataloaders=dataloader, return_predictions=False)


if __name__ == "__main__":
    main()
