#!/usr/bin/env python3
import argparse
import os
from itertools import groupby
from pathlib import Path
from shutil import copyfileobj
from typing import Optional, Union

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="concatenate protein embeddings and avg pool over the entire genome to create a genome embedding"
    )

    compress_args = parser.add_argument_group(
        "COMPRESSION"
    ).add_mutually_exclusive_group()

    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="basename of your embeddings files, will be appended by the -p/--pattern option",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="*.embeddings.txt",
        help="glob pattern to all the embeddings files (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--suffices",
        nargs=2,
        type=tuple,
        default=("embeddings.txt", "name_order.txt"),
        help="suffices for the output embedding and name order files for both the protein and genome embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=20,
        help="number of parallel threads used to read the concatenated embeddings file (default: %(default)s)",
    )

    compress_args.add_argument(
        "--parquet",
        action="store_true",
        help="compress embeddings files using parquet storage (default: %(default)s)",
    )

    # TODO: remove, very bad performance, esp for reading
    compress_args.add_argument(
        "--feather",
        action="store_true",
        help="compress embeddings files using feather storage (default: %(default)s)",
    )

    return parser.parse_args()


def get_embedding_files(pattern: str) -> list[Path]:
    files = sorted(
        Path.cwd().glob(pattern),
        key=lambda x: int(x.stem.rsplit(".")[0].rsplit("_", 1)[1]),
    )
    return files


def concat(files: list[Path], efile: str, nfile: str):
    # efile = "hq_viruses_cleaned.embeddings.txt"
    # nfile = "hq_viruses_cleaned.name_order.txt"
    with open(efile, "wb") as efp, open(nfile, "wb") as nfp:
        for file in files:
            namefile = f'{file.stem.rstrip("embeddings")}name_order.txt'

            for fsrcfile, fdst in zip((file, namefile), (efp, nfp)):
                with open(fsrcfile, "rb") as fsrc:
                    copyfileobj(fsrc, fdst)


def _read_embeddings(
    efile: str, to_numpy: bool = True
) -> Union["pl.DataFrame", np.ndarray]:
    data = pl.read_csv(efile, sep=" ", has_header=False)

    if to_numpy:
        return data.to_numpy()

    return data


def read_embeddings(efile: str) -> np.ndarray:
    return _read_embeddings(efile, True)


def read_embeddings_df(efile: str) -> "pl.DataFrame":
    return _read_embeddings(efile, False)


def genome_pooling(nfile: str, data: np.ndarray) -> dict[str, np.ndarray]:
    with open(nfile) as fp:
        nameorder = [(line.rstrip().rsplit("_", 1)[0], i) for i, line in enumerate(fp)]

    nameorder.sort()

    genome_embeddings = dict()
    groups = groupby(nameorder, key=lambda x: x[0])
    for genome, _indices in groups:
        indices = np.fromiter((x[1] for x in _indices), dtype=int)
        genome_emb = data[indices, :].mean(axis=0)
        genome_embeddings[genome] = genome_emb

    return genome_embeddings


def write_genome_embeddings(
    genome_embeddings: dict[str, np.ndarray], egfile: str, ngfile: str
):
    embed = np.array([v for v in genome_embeddings.values()])
    np.savetxt(egfile, embed)

    with open(ngfile, "w") as fp:
        for k in genome_embeddings.keys():
            fp.write(f"{k}\n")


def write_compressed(efile: str, egfile: str, filetype: str, compression: str = "zstd"):
    comp_dispatch = {
        "parquet": pl.DataFrame.write_parquet,
        "feather": pl.DataFrame.write_ipc,
    }

    writer = comp_dispatch[filetype]
    for file in (efile, egfile):
        outfile = f'{file.rsplit(".", 1)[0]}.{filetype}'
        if not os.path.exists(outfile):
            data = read_embeddings_df(file)
            writer(data, file=outfile, compression=compression)
        else:
            print(f"{outfile} already exists")


def main(
    basename: str,
    pattern: str,
    suffices: tuple[str, str],
    compressed_type: Optional[str] = None,
):
    namepattern = f"{basename}{pattern}"
    efile_suffix, nfile_suffix = suffices
    efile = f"{basename}.{efile_suffix}"
    nfile = f"{basename}.{nfile_suffix}"
    egfile = f"{basename}.genome.{efile_suffix}"
    ngfile = f"{basename}.genome.{nfile_suffix}"

    if not all(os.path.exists(f) for f in [efile, nfile, egfile, ngfile]):
        # can redo compression without having to redo these steps
        files = get_embedding_files(namepattern)
        concat(files, efile, nfile)
        data = read_embeddings(efile)
        genome_embeddings = genome_pooling(nfile, data)
        write_genome_embeddings(genome_embeddings, egfile, ngfile)
    else:
        print("Final embedding files already exist")

    if compressed_type is not None:
        write_compressed(efile, egfile, compressed_type)


if __name__ == "__main__":
    args = parse_args()

    threads = args.threads
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    import polars as pl

    compression = "parquet" if args.parquet else "feather" if args.feather else None

    main(
        basename=args.name,
        pattern=args.pattern,
        suffices=args.suffices,
        compressed_type=compression,
    )
