#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

import esm

MODELS = {
    "esm2_t48_15B": (esm.pretrained.esm2_t48_15B_UR50D, 48),
    "esm2_t6_8M": (esm.pretrained.esm2_t6_8M_UR50D, 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed protein sequences using ESM-2 from Meta AI"
    )

    required_args = parser.add_argument_group("REQUIRED")
    model_args = parser.add_argument_group("MODEL")

    required_args.add_argument(
        "-i", "--input", required=True, help="input protein fasta file"
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
        default=8192,
        type=int,
        help="number of tokens per batch (default: %(default)s)",
    )
    model_args.add_argument(
        "-th",
        "--torch-hub",
        default="/storage1/data14/esm/models/hub",
        help="path to the checkpoints/ directory with downloaded models (default: %(default)s)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        default="out",
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        default=30,
        type=int,
        help="number of pytorch threads (default: %(default)s)",
    )

    return parser.parse_args()


def main(
    seqfile: str,
    outdir: Path,
    model_name: str,
    batch_token_size: int,
    torch_hub: str,
    threads: int,
):
    outdir.mkdir(exist_ok=True)
    torch.set_num_threads(threads)
    torch.hub.set_dir(torch_hub)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_loader, layers = MODELS[model_name]
    model, alphabet = model_loader()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    seq_batches = esm.FastaBatchedDataset.from_file(seqfile)
    batch_indices = seq_batches.get_batch_indices(batch_token_size)

    embed_output = outdir.joinpath(f"{Path(seqfile).stem}.embeddings.txt")
    name_output = outdir.joinpath(f"{Path(seqfile).stem}.name_order.txt")

    with embed_output.open("ab") as efp, name_output.open("a") as nfp:
        for batch_idx in batch_indices:
            data = [seq_batches[idx] for idx in batch_idx]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                results = model(batch_tokens.to(device), repr_layers=[layers])

            token_repr = results["representations"][layers].to(torch.device("cpu"))

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            seq_repr = list()
            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(token_repr[i, 1 : tokens_len - 1].mean(0))

            # save results
            np.savetxt(efp, torch.stack(seq_repr).numpy())
            for label in batch_labels:
                nfp.write(f"{label}\n")


if __name__ == "__main__":
    args = parse_args()
    main(
        seqfile=args.input,
        outdir=Path(args.outdir),
        model_name=args.model,
        batch_token_size=args.batch_token_size,
        torch_hub=args.torch_hub,
        threads=args.threads,
    )
