from typing import Literal

import esm
import torch
import pytorch_lightning as pl

MODELS = {
    "esm2_t48_15B": esm.pretrained.esm2_t48_15B_UR50D,
    "esm2_t36_3B": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_t33_650M": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_t30_150M": esm.pretrained.esm2_t30_150M_UR50D,
    "esm2_t12_35M": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_t6_8M": esm.pretrained.esm2_t6_8M_UR50D,
}
MODELVALUES = Literal[
    "esm2_t48_15B",
    "esm2_t36_3B",
    "esm2_t33_650M",
    "esm2_t30_150M",
    "esm2_t12_35M",
    "esm2_t6_8M",
]

BatchType = tuple[list[str], list[str], torch.Tensor]
# Prediction only
class ESM2(pl.LightningModule):
    def __init__(self, model: esm.ESM2, alphabet: esm.Alphabet) -> None:
        super().__init__()
        self.model = model
        self.alphabet = alphabet
        self.repr_layers = model.num_layers

    def predict_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        labels, seqs, tokens = batch
        seqlens = torch.sum(tokens != self.alphabet.padding_idx, dim=1)
        results = self.model(
            tokens, repr_layers=[self.repr_layers], return_contacts=False
        )
        token_repr: torch.Tensor = (
            results["representations"][self.repr_layers].cpu().detach()
        )

        # Generate sequence level representations by averaging over token repr
        # NOTE: token 0 is beginning-of-seq token
        seq_rep = torch.vstack(
            [
                token_repr[i, 1 : token_lens - 1].mean(dim=0)
                for i, token_lens in enumerate(seqlens)
            ]
        )
        return seq_rep

    @classmethod
    def from_model_name(cls, model_name: MODELVALUES) -> "ESM2":
        model_loader = MODELS[model_name]
        esm_model: esm.ESM2
        alphabet: esm.Alphabet
        esm_model, alphabet = model_loader()
        return cls(model=esm_model, alphabet=alphabet)
