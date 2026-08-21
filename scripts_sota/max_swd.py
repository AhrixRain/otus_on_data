"""Adversarially maximized sliced-Wasserstein slicing (max-SW).

Random slicing directions are the principal weakness of the 2021 SWAE
objective for narrow physics features: most directions are blind to a
30 MeV resonance shell.  Instead of hand-picking physics directions, this
module keeps a small set of learned directions that are adversarially updated
to *maximize* the sliced-Wasserstein distance on the current minibatch.  The
generative model then minimizes the same distance at the updated directions.

Usage sketch::

    max_swd = MaxSlicedWasserstein(feature_dim=8, num_directions=32, lr=1e-3)
    for x, z in loader:
        max_swd.adversarial_step(z, z_encoded)   # update directions only
        loss = base_loss + weight * max_swd(z, z_encoded)
        loss.backward()
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaxSlicedWasserstein(nn.Module):
    """Learned slicing directions for a sliced p-Wasserstein distance.

    The directions are normalized rows of ``self.directions``.  A cosine
    diversity penalty keeps them from collapsing to a single projection.

    Stability controls (added after the Run E stage-2 explosion):

    * ``p=1`` is preferred: squared sorted differences are unbounded and
      chase stochastic decoder noise.
    * ``distance_clamp`` saturates the value fed to the generative loss; the
      adversarial direction ascent still sees the raw distance.
    * ``direction_grad_clip`` bounds each direction update (the model-side
      gradient was clipped before, but the adversarial ascent was not).
    * ``update_every`` updates directions only every N distribution-loss
      calls, so they cannot chase every minibatch's fresh noise draw.
    """

    def __init__(
        self,
        feature_dim: int,
        num_directions: int = 32,
        p: int = 2,
        lr: float = 1e-3,
        diversity_weight: float = 0.05,
        eps: float = 1e-8,
        direction_grad_clip: float = 1.0,
        update_every: int = 5,
        distance_clamp: float | None = None,
    ):
        super().__init__()
        if feature_dim <= 0 or num_directions <= 0:
            raise ValueError("feature_dim and num_directions must be positive")
        if int(p) not in {1, 2}:
            raise ValueError("Only p=1 and p=2 are supported")
        self.p = int(p)
        self.lr = float(lr)
        self.diversity_weight = float(diversity_weight)
        self.eps = float(eps)
        self.direction_grad_clip = float(direction_grad_clip)
        self.update_every = max(1, int(update_every))
        self.distance_clamp = None if distance_clamp is None else float(distance_clamp)
        if self.distance_clamp is not None and self.distance_clamp <= 0.0:
            raise ValueError("distance_clamp must be positive")
        directions = torch.randn(num_directions, feature_dim)
        directions = F.normalize(directions, dim=1, eps=eps)
        self.directions = nn.Parameter(directions, requires_grad=True)
        self.register_buffer("_call_count", torch.zeros((), dtype=torch.long))

    def normalized_directions(self) -> torch.Tensor:
        return F.normalize(self.directions, dim=1, eps=self.eps)

    def project(self, values: torch.Tensor) -> torch.Tensor:
        """Project [N, D] onto the learned directions -> [N, L]."""
        if values.ndim != 2:
            raise ValueError("max-SW expects a rank-2 tensor")
        return values @ self.normalized_directions().t()

    def distance(self, a: torch.Tensor, b: torch.Tensor, clamp: bool = False) -> torch.Tensor:
        """Sliced p-Wasserstein distance at the current directions."""
        if a.shape != b.shape or a.shape[1] != self.directions.shape[1]:
            raise ValueError(
                f"max-SW requires equal-shaped batches in the same feature space; "
                f"got {tuple(a.shape)} and {tuple(b.shape)}"
            )
        pa = torch.sort(self.project(a), dim=0).values
        pb = torch.sort(self.project(b), dim=0).values
        diff = pa - pb
        if self.p == 1:
            value = diff.abs().mean()
        else:
            value = diff.square().mean()
        if clamp and self.distance_clamp is not None:
            value = value.clamp(max=self.distance_clamp)
        return value

    def diversity_penalty(self) -> torch.Tensor:
        directions = self.normalized_directions()
        gram = directions @ directions.t()
        off_diag = gram - torch.diag(gram).diag_embed()
        return off_diag.abs().mean()

    def adversarial_step(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
        """One gradient-*ascent* step on the slicing directions.

        Only ``self.directions`` is updated; the inputs are detached from the
        computational graph so the generative model receives no gradient here.
        """
        self.directions.requires_grad_(True)
        a_det = a.detach()
        b_det = b.detach()
        raw_distance = self.distance(a_det, b_det, clamp=False)
        self._call_count += 1
        should_update = bool((self._call_count % self.update_every) == 0)
        direction_grad_norm = 0.0
        if should_update:
            objective = raw_distance - self.diversity_weight * self.diversity_penalty()
            grad = torch.autograd.grad(objective, self.directions, retain_graph=False)[0]
            if grad is not None:
                grad_norm = grad.norm()
                direction_grad_norm = float(grad_norm.detach())
                if grad_norm > self.direction_grad_clip:
                    grad = grad * (self.direction_grad_clip / grad_norm)
                with torch.no_grad():
                    self.directions.data.add_(self.lr * grad)
                    self.directions.data.copy_(
                        F.normalize(self.directions.data, dim=1, eps=self.eps)
                    )
        return {
            "max_swd_raw": float(raw_distance.detach()),
            "direction_grad_norm": direction_grad_norm,
            "direction_updated": should_update,
            "direction_diversity": float(self.diversity_penalty().detach()),
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.distance(a, b, clamp=True)
