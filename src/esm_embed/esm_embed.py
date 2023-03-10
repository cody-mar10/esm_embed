#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import esm
import pytorch_lightning as pl
import torch
from pytorch_lightning.accelerators import find_usable_cuda_devices
from torch.utils.data import DataLoader

from esm_embed import arch
from esm_embed.arch.model import MODELS, MODELVALUES


@dataclass
class Args:
    input: Path
    model: MODELVALUES
    batch_size: int
    torch_hub: Path
    outdir: Path
    devices: int
    accelerator: Literal["cpu", "gpu", "auto"]


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Embed protein sequences using ESM-2 from Meta AI"
    )

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
        default=1,
        type=int,
        help="number of cpus/gpus to use. If using cpus, this will convert to threads on a single cpu node. (default: %(default)s)",
    )
    parser.add_argument(
        "-a",
        "--accelerator",
        metavar="DEVICE",
        default="auto",
        choices={"cpu", "gpu", "auto"},
        help="type of device to use (default: %(default)s) [choices: %(choices)s]",
    )

    args = parser.parse_args()
    return Args(
        input=args.input,
        model=args.model,
        batch_size=args.batch_token_size,
        torch_hub=args.torch_hub,
        outdir=args.outdir,
        devices=args.devices,
        accelerator=args.accelerator,
    )


# PREDICT MAIN
# TODO: make one for concatenating results
def main():
    args = parse_args()
    torch.hub.set_dir(args.torch_hub)
    pl.seed_everything(111)

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
        parallelism_kwargs = {"devices": 1}
    elif args.accelerator == "gpu":
        parallelism_kwargs = {"devices": find_usable_cuda_devices(args.devices)}
    else:
        parallelism_kwargs = {"devices": "auto"}

    trainer = pl.Trainer(
        enable_checkpointing=False,
        callbacks=[writer],
        accelerator=args.accelerator,
        logger=False,
        **parallelism_kwargs,
    )

    trainer.predict(model=model, dataloaders=dataloader, return_predictions=False)


if __name__ == "__main__":
    main()
