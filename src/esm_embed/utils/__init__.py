from __future__ import annotations
import argparse
import sys
from pathlib import Path

from tables import Filters


def sort_key(file: Path) -> int:
    """Return the trailing number on a file path if it exists

    Args:
        file (Path): any file path
    """
    return int(file.name.split(".")[0].rsplit("_", 1)[1])


COMPRESSION_FILTER = Filters(complevel=4, complib="blosc:lz4")


class _HelpAction(argparse._HelpAction):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string=None,
    ):
        parser.print_help()

        # retrieve subparsers from parser
        subparsers_actions = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        for subparsers_action in subparsers_actions:
            for choice, subparser in subparsers_action.choices.items():
                print(f"{choice:10s}{subparser.description}")

        parser.exit()


def _skip_prgm_and_command() -> list[str]:
    if sys.argv[0].startswith("python"):
        # 0 = python
        # 1 = PATH/TO/esm
        # 2 = embed/concat
        return sys.argv[3:]
    # wrapper cli script
    elif sys.argv[0].endswith("esm"):
        # 0 = PATH/TO/esm
        # 1 = embed/concat
        return sys.argv[2:]
    else:
        return sys.argv[1:]
