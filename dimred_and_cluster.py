#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import hdbscan
import numba
import umap
import pandas as pd
from numpy import ndarray
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
    names: Path
    output: Path
    protein: bool
    threads: int
    no_stand: bool
    pca_args: PcaArgs
    umap_args: UmapArgs
    hdbscan_args: HdbscanArgs


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
        help="input genome embeddings",
    )
    parser.add_argument(
        "-n",
        "--names",
        metavar="FILE",
        required=True,
        type=Path,
        help="genome names associated with embeddings",
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
        "-m",
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
        names=args.names,
        output=args.output,
        protein=args.protein,
        threads=args.threads,
        no_stand=args.no_stand,
        pca_args=pca_args,
        umap_args=umap_args,
        hdbscan_args=hdbscan_args,
    )


def read_data(file: Path, no_stand: bool) -> ndarray:
    import polars as pl

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
    X: ndarray,
    controller: ThreadpoolController,
    threads: int,
    int_components: int,
    n_neighbors: int,
    umap_components: int,
    dens_lambda: float,
    metric: str,
    no_pca: bool,
    low_memory: bool,
) -> ndarray:
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
            X = pca.fit_transform(X)
        return cast(ndarray, umapper.fit_transform(X))


def cluster(
    X: ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
) -> hdbscan.HDBSCAN:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    return clusterer.fit(X)


def main():
    args = parse_args()
    threads = args.threads
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    numba.set_num_threads(threads)
    controller = ThreadpoolController()

    X = read_data(args.input, args.no_stand)
    if args.protein:
        low_memory = True
        names = pd.read_csv(args.names, names=["protein"])
    else:
        low_memory = False
        names = pd.read_csv(args.names, names=["genome"])

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

    (
        pd.DataFrame(X, columns=[f"UMAP{i+1}" for i in range(X.shape[1])])
        .assign(cluster=clusters)
        .astype({"cluster": "category"})
        .merge(names, left_index=True, right_index=True)
        .to_csv(args.output, sep="\t", index=False)
    )


if __name__ == "__main__":
    main()
