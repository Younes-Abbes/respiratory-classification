"""
Sharpness-Aware Minimization (SAM) and Adaptive SAM (ASAM) optimizers.

Reference:
    Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
    Generalization" (ICLR 2021).
    Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for
    Scale-Invariant Learning of Deep Neural Networks" (ICML 2021).

SAM is a two-step optimizer:
    1. Compute gradient g at current weights w.
    2. Perturb weights to w + e where e = rho * g / ||g||  (climb to sharpest point).
    3. Compute gradient at the perturbed weights.
    4. Restore original weights and apply the perturbed gradient with the base optimizer.

ASAM rescales the perturbation per-parameter by |w| so that flat directions
in well-conditioned coordinates aren't penalised. It typically allows a
larger rho and gives better results on imbalanced data.
"""

from typing import Iterable, Type

import torch


class SAM(torch.optim.Optimizer):
    """Vanilla Sharpness-Aware Minimization.

    Wrap any base PyTorch optimizer (AdamW, SGD, ...) so that each step
    becomes a min-max update toward flat minima.

    Usage:
        base_opt = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_opt, lr=1e-5, rho=0.05,
                        weight_decay=1e-4)

        # In the training loop:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)        # climb to perturbed point
        criterion(model(x), y).backward()           # second forward/backward
        optimizer.second_step(zero_grad=True)       # restore + apply update
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: Type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # Re-build the base optimizer over our parameter groups so its state
        # is shared with this wrapper.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Climb to w + e* where e* maximises the loss in a rho-ball."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Save the original parameters so we can restore them later.
                self.state[p]["old_p"] = p.data.clone()

                if group["adaptive"]:
                    # ASAM: scale the perturbation by |w| so that scale-invariance
                    # is preserved across re-parameterisations.
                    e_w = (torch.pow(p, 2)) * p.grad * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore w from w + e* and run the base optimizer."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]  # back to "real" weights
        # Apply the base optimizer step using the gradients computed at w + e*.
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """A closure-based step for full SAM if needed."""
        if closure is None:
            raise RuntimeError(
                "SAM requires a closure that re-evaluates the loss; "
                "use first_step / second_step explicitly instead."
            )
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        # We use the norm in the parameter space across all groups.
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    (
                        (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                    ).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):  # type: ignore[override]
        super().load_state_dict(state_dict)
        # Make sure the base optimizer's param_groups stay synchronised.
        self.base_optimizer.param_groups = self.param_groups
