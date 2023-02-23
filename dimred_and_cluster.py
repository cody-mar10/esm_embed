#!/usr/bin/env python3
import argparse
import os

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="generate 2D genome embeddings from ESM2 genome-pooled embeddings"
    )

    pca_args = parser.add_argument_group("PCA ARGS")
    umap_args = parser.add_argument_group("UMAP ARGS")
    hdbscan_args = parser.add_argument_group("HDBSCAN ARGS")

    parser.add_argument("-i", "--input", required=True, help="input genome embeddings")
    parser.add_argument(
        "-n", "--names", required=True, help="genome names associated with embeddings"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="clusters.tsv",
        help="output table name (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=15,
        help="max number of threads to use (default: %(default)s)",
    )
    parser.add_argument(
        "--no-stand",
        action="store_true",
        help="use to skip the standardization step before dimensionality reduction (default: %(default)s)",
    )

    pca_args.add_argument(
        "-c",
        "--n-components",
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
        type=int,
        default=15,
        help="number of neighbors for UMAP kNN graph (default: %(default)s)",
    )
    umap_args.add_argument(
        "-d",
        "--density-lambda",
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
        metavar="COMP",
        type=int,
        default=2,
        help="number of UMAP components (default: %(default)s) [choices: %(choices)s]",
    )

    hdbscan_args.add_argument(
        "-e",
        "--eps",
        type=float,
        default=0.0,
        help="DBSCAN eps parameter / hDBSCAN cluster_selection_epsilon (default: %(default)s)",
    )
    hdbscan_args.add_argument(
        "-mc",
        "--min-cluster-size",
        type=int,
        default=2,
        help="min number of points to define a cluster (default: %(default)s)",
    )
    hdbscan_args.add_argument(
        "-ms",
        "--min-samples",
        type=int,
        default=1,
        help="min number of neighbors to a point (default: %(default)s)",
    )

    return parser.parse_args()


def read_data(file: str, no_stand: bool) -> ndarray:
    ext = file.rsplit(".", 1)[1]

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
) -> ndarray:
    pca = PCA(n_components=int_components, random_state=SEED)
    umapper = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=umap_components,
        densmap=True,
        dens_lambda=dens_lambda,
        metric=metric,
        random_state=SEED,
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


def main(
    embeddingsfile: str,
    namesfile: str,
    no_stand: bool,
    output: str,
    threads: int,
    n_components: int,
    n_neighbors: int,
    umap_components: int,
    dens_lambda: float,
    metric: str,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
    no_pca: bool,
):
    numba.set_num_threads(threads)
    controller = ThreadpoolController()
    X = read_data(embeddingsfile, no_stand)
    names = pd.read_csv(namesfile, names=["genome"])

    X = dimred(
        X,
        controller,
        threads,
        n_components,
        n_neighbors,
        umap_components,
        dens_lambda,
        metric,
        no_pca,
    )
    clusters = cluster(
        X, min_cluster_size, min_samples, cluster_selection_epsilon
    ).labels_

    (
        pd.DataFrame(X, columns=[f"UMAP{i+1}" for i in range(X.shape[1])])
        .assign(cluster=clusters)
        .astype({"cluster": "category"})
        .merge(names, left_index=True, right_index=True)
        .to_csv(output, sep="\t", index=False)
    )


if __name__ == "__main__":
    args = parse_args()

    threads = args.threads

    os.environ["POLARS_MAX_THREADS"] = str(threads)
    import polars as pl

    main(
        embeddingsfile=args.input,
        namesfile=args.names,
        no_stand=args.no_stand,
        output=args.output,
        threads=threads,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        umap_components=args.umap_components,
        dens_lambda=args.density_lambda,
        metric=args.distance_metric,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.eps,
        no_pca=args.no_pca,
    )
