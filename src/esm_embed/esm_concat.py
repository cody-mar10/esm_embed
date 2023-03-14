#!/usr/bin/env python3
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfileobj

import numpy as np
import tables as tb
from numpy.typing import NDArray

from .utils import sort_key, COMPRESSION_FILTER, _skip_prgm_and_command


@dataclass
class Args:
    directories: list[Path]
    data_output: Path
    name_output: Path

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace):
        return cls(
            directories=namespace.directories,
            data_output=namespace.data_output,
            name_output=namespace.name_output,
        )


def _add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-d",
        "--directories",
        metavar="DIR",
        nargs="+",
        type=Path,
        required=True,
        help="ESM2 embeddings directories produced by esm_embed",
    )
    parser.add_argument(
        "-o",
        "--data-output",
        metavar="FILE",
        type=Path,
        default=Path("embeddings.h5"),
        help="name of concatenated embeddings file (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--name-output",
        metavar="FILE",
        type=Path,
        default=Path("names.txt"),
        help="name of metadata file with names of proteins (default: %(default)s)",
    )


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Concatenate ESM2 embeddings")
    _add_args(parser)
    args = parser.parse_args(_skip_prgm_and_command())
    return Args(
        directories=args.directories,
        data_output=args.data_output,
        name_output=args.name_output,
    )


def read_all_data(files: list[Path]) -> NDArray[np.float32]:
    data: list[NDArray[np.float32]] = list()
    for file in files:
        with tb.File(file) as fp:
            data.append(fp.root.data[:])
    return np.vstack(data)


def concat_data(data: NDArray[np.float32], output: Path):
    with tb.File(output, "w") as fp:
        fp.create_carray(
            "/", "data", obj=data, shape=data.shape, filters=COMPRESSION_FILTER
        )


def concat_names(files: list[Path], output: Path):
    with output.open("wb") as fdst:
        for file in files:
            with file.open("rb") as fsrc:
                copyfileobj(fsrc, fdst)


def main():
    args = parse_args()

    data_inputs: list[Path] = list()
    name_inputs: list[Path] = list()
    for directory in args.directories:
        data_inputs.extend(directory.glob("*.h5"))
        name_inputs.extend(directory.glob("*.names.txt"))

    data_inputs.sort(key=sort_key)
    name_inputs.sort(key=sort_key)

    concat_names(name_inputs, args.name_output)
    data = read_all_data(data_inputs)
    concat_data(data, args.data_output)


if __name__ == "__main__":
    main()
