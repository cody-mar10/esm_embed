#!/usr/bin/env python3
import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import hdbscan
import numba
import umap
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import ThreadpoolController
from typing import cast

SEED = 111


@dataclass
class PcaArgs:
    n_components: int
    no_pca: bool


@dataclass
class UmapArgs:
    n_neighbors: int
    dens_lambda: float
    metric: str
    n_components: int


@dataclass
class HdbscanArgs:
    eps: float
    min_cluster_size: int
    min_samples: int


@dataclass
class Args:
    input: Path
    model: str
    output: Path
    log: Path
    protein: bool
    threads: int
    no_stand: bool
    pca_args: PcaArgs
    umap_args: UmapArgs
    hdbscan_args: HdbscanArgs


MODELS = {
    "esm2_t48_15B",
    "esm2_t36_3B",
    "esm2_t33_650M",
    "esm2_t30_150M",
    "esm2_t12_35M",
    "esm2_t6_8M",
}


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="generate 2D genome embeddings from ESM2 genome-pooled embeddings"
    )

    pca_args = parser.add_argument_group("PCA ARGS")
    umap_args = parser.add_argument_group("UMAP ARGS")
    hdbscan_args = parser.add_argument_group("HDBSCAN ARGS")

    parser.add_argument(
        "-i",
        "--input",
        metavar="FILE",
        required=True,
        type=Path,
        help="hdf5 embeddings file",
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        default="esm2_t33_650M",
        choices=MODELS,
        help="esm model to use (default: %(default)s) [choices: %(choices)s]",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default="clusters.tsv",
        type=Path,
        help="output table name (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--log",
        metavar="FILE",
        default="dimred.log",
        type=Path,
        help="log file name (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        metavar="INT",
        type=int,
        default=15,
        help="max number of threads to use (default: %(default)s)",
    )
    parser.add_argument(
        "--no-stand",
        action="store_true",
        help="use to skip the standardization step before dimensionality reduction (default: %(default)s)",
    )
    parser.add_argument(
        "--protein",
        action="store_true",
        help="use if working with protein embeddings instead of genome embeddings(default: %(default)s)",
    )

    pca_args.add_argument(
        "-c",
        "--n-components",
        metavar="INT",
        type=int,
        default=20,
        help="number of PCs to chose -- good strategy is to choose as few as needed to account for 90%% of variation (default: %(default)s)",
    )
    pca_args.add_argument(
        "--no-pca",
        action="store_true",
        help="use to skip pca step and use UMAP to embed directly from original high dimensional space",
    )

    umap_args.add_argument(
        "-nn",
        "--n-neighbors",
        metavar="INT",
        type=int,
        default=15,
        help="number of neighbors for UMAP kNN graph (default: %(default)s)",
    )
    umap_args.add_argument(
        "-d",
        "--density-lambda",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="density-regularized UMAP regularization coefficient (default: %(default)s)",
    )
    umap_args.add_argument(
        "-dm",
        "--distance-metric",
        metavar="METRIC",
        choices={"euclidean", "manhattan", "correlation", "cosine", "chebyshev"},
        default="euclidean",
        help="distance metric (default: %(default)s) [choices: %(choices)s]",
    )
    umap_args.add_argument(
        "-u",
        "--umap-components",
        metavar="INT",
        type=int,
        default=2,
        help="number of UMAP components (default: %(default)s)",
    )

    hdbscan_args.add_argument(
        "-e",
        "--eps",
        metavar="FLOAT",
        type=float,
        default=0.0,
        help="DBSCAN eps parameter / hDBSCAN cluster_selection_epsilon (default: %(default)s)",
    )
    hdbscan_args.add_argument(
        "-mc",
        "--min-cluster-size",
        metavar="INT",
        type=int,
        default=2,
        help="min number of points to define a cluster (default: %(default)s)",
    )
    hdbscan_args.add_argument(
        "-ms",
        "--min-samples",
        metavar="INT",
        type=int,
        default=1,
        help="min number of neighbors to a point (default: %(default)s)",
    )

    args = parser.parse_args()

    pca_args = PcaArgs(n_components=args.n_components, no_pca=args.no_pca)
    umap_args = UmapArgs(
        n_neighbors=args.n_neighbors,
        dens_lambda=args.density_lambda,
        metric=args.distance_metric,
        n_components=args.umap_components,
    )
    hdbscan_args = HdbscanArgs(
        eps=args.eps,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    return Args(
        input=args.input,
        model=args.model,
        output=args.output,
        log=args.log,
        protein=args.protein,
        threads=args.threads,
        no_stand=args.no_stand,
        pca_args=pca_args,
        umap_args=umap_args,
        hdbscan_args=hdbscan_args,
    )


def read_data(file: Path, no_stand: bool, threads: int) -> np.ndarray:
    # DEPRECATED
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    import polars as pl

    logging.debug(f"Polars using {pl.threadpool_size()} threads")

    ext = file.suffix.lstrip(".")

    if ext == "txt":
        X = pl.read_csv(file, sep=" ", has_header=False).to_numpy()
    elif ext == "parquet":
        X = pl.read_parquet(file).to_numpy()
    else:
        raise RuntimeError(
            f"Embeddings file format {ext} not allowed. You can only supply raw .txt or .parquet files."
        )
    return X if no_stand else StandardScaler().fit_transform(X)


def dimred(
    X: h5py.Dataset,
    controller: ThreadpoolController,
    threads: int,
    int_components: int,
    n_neighbors: int,
    umap_components: int,
    dens_lambda: float,
    metric: str,
    no_pca: bool,
    low_memory: bool,
) -> np.ndarray:
    pca = PCA(n_components=int_components, random_state=SEED)
    umapper = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=umap_components,
        densmap=True,
        dens_lambda=dens_lambda,
        metric=metric,
        random_state=SEED,
        low_memory=low_memory,
    )

    with controller.limit(limits=threads):
        if not no_pca:
            logging.info("Running PCA")
            X = pca.fit_transform(X)
        logging.info("Running UMAP")
        return cast(np.ndarray, umapper.fit_transform(X))


def cluster(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
) -> hdbscan.HDBSCAN:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    logging.info("Running clustering with HDBSCAN")
    return clusterer.fit(X)


def main():
    args = parse_args()
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"Arguments: {args}")

    threads = args.threads
    numba.set_num_threads(threads)
    controller = ThreadpoolController()

    logging.info(f"Reading {args.input} with {threads} threads")

    if args.protein:
        low_memory = True
        data_type = "protein"
    else:
        low_memory = False
        data_type = "genome"

    base = args.input.stem.split(".")[0]
    embed_hdf5_key = f"{base}/embeddings/{args.model}/{data_type}"
    names_hdf5_key = f"{base}/names/{data_type}"
    with h5py.File(args.input) as h5fp:
        logging.info(f"Reading {embed_hdf5_key} from {args.input}")
        if args.no_stand:
            X = h5fp[embed_hdf5_key]
        else:
            with controller.limit(limits=threads):
                X = StandardScaler().fit_transform(h5fp[embed_hdf5_key])

        logging.info(f"Reading {names_hdf5_key} from {args.input}")
        names = pd.DataFrame(
            np.array(h5fp[names_hdf5_key]).astype(str), columns=[data_type]
        )

        logging.info("Performing dimensionality reduction.")
        X = dimred(
            X,
            controller,
            threads,
            args.pca_args.n_components,
            args.umap_args.n_neighbors,
            args.umap_args.n_components,
            args.umap_args.dens_lambda,
            args.umap_args.metric,
            args.pca_args.no_pca,
            low_memory,
        )
        clusters = cluster(
            X,
            args.hdbscan_args.min_cluster_size,
            args.hdbscan_args.min_samples,
            args.hdbscan_args.eps,
        ).labels_

        outdir = args.input.parent.joinpath(f"{args.model}_results")
        output = outdir.joinpath(args.output)
        logging.info(f"Saving output to {output}")
        (
            pd.DataFrame(X, columns=[f"UMAP{i+1}" for i in range(X.shape[1])])
            .assign(cluster=clusters)
            .astype({"cluster": "category"})
            .merge(names, left_index=True, right_index=True)
            .to_csv(args.output, sep="\t", index=False)
        )


if __name__ == "__main__":
    main()
