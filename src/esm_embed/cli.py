#!/usr/bin/env python3
import argparse
from enum import Enum, auto
from typing import Literal, Union

from . import esm_concat, esm_embed
from .utils import _HelpAction


class Mode:
    EMBED = auto()
    CONCAT = auto()


def parse_args() -> int:
    parser = argparse.ArgumentParser(
        "ESM utilities for inference and post processing", add_help=False
    )
    parser.add_argument(
        "-h", "--help", action=_HelpAction, help="show this help message and exit"
    )

    # only required to display help
    # the actual commands will take care of their own parsing
    subparsers = parser.add_subparsers(help="COMMANDS", dest="command")
    embed_parser = subparsers.add_parser(
        "embed", description="Generate ESM2 embeddings"
    )
    esm_embed._add_args(embed_parser)
    concat_parser = subparsers.add_parser(
        "concat", description="Concat ESM2 embeddings"
    )
    esm_concat._add_args(concat_parser)

    args = parser.parse_args()

    match args.command:
        case "embed":
            return Mode.EMBED
        case "concat":
            return Mode.CONCAT
        case _:
            raise ValueError("Invalid command chosen. Choose from: [embed, concat]")


def main():
    command = parse_args()

    match command:
        case Mode.EMBED:
            esm_embed.main()
        case Mode.CONCAT:
            esm_concat.main()


if __name__ == "__main__":
    main()
