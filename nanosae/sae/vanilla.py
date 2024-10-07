import einops
import torch.nn as nn
import torch
import torch.nn.functional as F

from jaxtyping import Float
from torch import Tensor

from nanosae.core import SAE, SAETrainingWrapper
from nanosae.utils.device import get_device

class VanillaSAE(SAE):
    d_in: int
    d_sae: int

    W_enc: Float[Tensor, "d_in d_sae"]
    _W_dec: Float[Tensor, "d_sae d_in"] | None
    b_enc: Float[Tensor, " d_sae"]
    b_dec: Float[Tensor, " d_in"]

    def __init__(
        self,
        n_inst: int,
        d_in: int,
        d_sae: int,
        weight_normalize_eps: float = 1e-8,
        tied_weights: bool = False,
        device=None,
    ):
        if device is None:
            device = get_device()
        super(SAE, self).__init__()
        self.n_inst = n_inst
        self.d_in = d_in
        self.d_sae = d_sae
        self.weight_normalize_eps = weight_normalize_eps
        self.tied_weights = tied_weights

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_sae)))
        )
        self._W_dec = (
            None
            if tied_weights
            else nn.Parameter(
                nn.init.kaiming_uniform_(torch.empty((n_inst, d_sae, d_in)))
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(n_inst, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(n_inst, d_in))

        self.to(device)

    @property
    def W_dec(self) -> Float[Tensor, "inst d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-1, -2)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "inst d_sae d_in"]:
        """Returns decoder weights, normalized over the autoencoder input dimension."""
        return self.W_dec / (self.W_dec.norm(dim=-1, keepdim=True) + self.weight_normalize_eps)

    def encode(
        self, x: Float[Tensor, "... d_in"]
    ) -> Float[Tensor, "... d_sae"]:
        # Center the input
        x_cent = x - self.b_dec
        pre_acts = (
            einops.einsum(
                x_cent, self.W_enc, "... d_in, d_in d_sae -> ... d_sae"
            )
            + self.b_enc
        )
        return F.relu(pre_acts)

    def decode(
        self, z: Float[Tensor, "... d_sae"]
    ) -> Float[Tensor, "... d_in"]:
        return (
            einops.einsum(
                z,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )


class VanillaSAETrainingWrapper(SAETrainingWrapper):
    def __init__(self, l1_coeff: float = 1e-3):
        self.l1_coeff = l1_coeff

    def training_forward_pass(
        self, sae: VanillaSAE, x: Float[Tensor, "... d_in"]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        h = sae.encode(x)
        h_recon = sae.decode(h)

        L_reconstruction = (h_recon - h).pow(2).mean(-1)
        L_sparsity = h.abs().sum(-1)
        info_dict = {
            "L_reconstruction": L_reconstruction,
            "L_sparsity": L_sparsity,
        }
        loss = (L_reconstruction + self.l1_coeff * L_sparsity).mean(0).sum()
        return loss, info_dict