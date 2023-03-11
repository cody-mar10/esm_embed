from __future__ import annotations
from pathlib import Path
from shutil import copyfileobj, rmtree
from typing import Callable, Literal, Optional

import numpy as np
import lightning as L
import tables as tb
import torch
from numpy.typing import NDArray
from lightning.pytorch.callbacks import BasePredictionWriter

from .model import BatchType, ESM2

LAYERS_TO_MODELNAME = {
    6: "esm2_t6_8M",
    48: "esm2_t48_15B",
    36: "esm2_t36_3B",
    33: "esm2_t33_650M",
    30: "esm2_t30_150M",
    12: "esm2_t12_35M",
}


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        outdir: Path,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.outdir = outdir
        self.compression = tb.Filters(complib="blosc:lz4", complevel=4)
        self.dataset_prefix = "dataset"
        self.batch_prefix = "batch"

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: ESM2,
        predictions: torch.Tensor,
        batch_indices: Optional[list[int]],
        batch: BatchType,
        batch_idx: int,
        dataloader_idx: int,
    ):
        names = batch[0]
        model_name = LAYERS_TO_MODELNAME[pl_module.model.num_layers]
        outdir = self.outdir.joinpath(
            f"{model_name}_{self.dataset_prefix}_{dataloader_idx}"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        output_file = outdir.joinpath(
            f"{model_name}_{self.batch_prefix}_{batch_idx}.h5"
        )
        name_output = outdir.joinpath(
            f"{model_name}_{self.batch_prefix}_{batch_idx}.names.txt"
        )
        predictions = predictions.numpy()
        with tb.File(output_file, "w") as fp:
            fp.create_carray(
                "/",
                "data",
                obj=predictions,
                shape=predictions.shape,
                filters=self.compression,
            )

        with name_output.open("w") as fp:
            for name in names:
                fp.write(f"{name}\n")

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Combine all separate batch files into a single file

        Args:
            trainer (pl.Trainer): pytorch-lightning trainer
            pl_module (pl.LightningModule): pytorch-lightning module
        """
        super().on_predict_end(trainer, pl_module)

        sortkey: Callable[[Path], int] = lambda x: int(
            x.name.split(".")[0].rsplit("_", 1)[1]
        )
        dataset_paths = sorted(
            self.outdir.glob(f"*{self.dataset_prefix}*/"), key=sortkey
        )
        data_paths = {
            dataset_path: sorted(
                dataset_path.glob(f"*{self.batch_prefix}*.h5"), key=sortkey
            )
            for dataset_path in dataset_paths
        }
        name_paths = {
            dataset_path: sorted(
                dataset_path.glob(f"*{self.batch_prefix}*.txt"), key=sortkey
            )
            for dataset_path in dataset_paths
        }

        # copy .h5 embeddings arrays
        for dataset_path, batch_paths in data_paths.items():
            data_output = self.outdir.joinpath(f"{dataset_path.name}.h5")
            data: list[NDArray[np.float32]] = list()
            with tb.File(data_output, "w") as fdst:
                for data_path in batch_paths:
                    with tb.File(data_path) as fsrc:
                        data.append(fsrc.root.data[:])

                concat = np.vstack(data)
                fdst.create_carray(
                    "/",
                    "data",
                    obj=concat,
                    shape=concat.shape,
                    filters=self.compression,
                )

        # copy sequence names
        for dataset_path, batch_paths in name_paths.items():
            data_output = self.outdir.joinpath(f"{dataset_path.name}.names.txt")
            with data_output.open("wb") as fdst:
                for name_path in batch_paths:
                    with name_path.open("rb") as fsrc:
                        copyfileobj(fsrc, fdst)

        # delete dataset files
        for dataset_path in dataset_paths:
            rmtree(dataset_path)
