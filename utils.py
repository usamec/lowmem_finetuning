import torch
from typing import Iterable, Optional, Union
from torch.optim import Optimizer

def add_scaled_bf16(A: torch.Tensor, B: torch.Tensor, s) -> torch.Tensor:
    """
    Return (A + s*B) in bf16 with some stochastic rounding.
    A, B : bf16 tensors of the same shape / device
    s    : scalar (Python number or 0-D tensor)
    """
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise TypeError("A and B must be bfloat16 tensors")
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    move = s * B
    x = A + move
    real_move = x - A
    x_closer = x
    error = real_move - move
    x_farther = torch.nextafter(x_closer, torch.where(error>0, -torch.inf, torch.inf).to(torch.bfloat16))
    
    ulp = torch.abs(x_farther - x_closer)
    rand_unif = torch.rand_like(x)
    use_farther = rand_unif * ulp < torch.abs(error)
    x_stoch = torch.where(use_farther, x_farther, x_closer)

    assert x_stoch.dtype == torch.bfloat16
    return x_stoch

class Adafactor(Optimizer):
    r"""
    Simplified Adafactor (factorised second moment only + RMS scaling).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        beta2: float = 0.99,
        eps: Union[float, tuple[float, float]] = (1e-30, 1e-30),
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not 0.0 < beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")

        # allow eps as single float or pair
        if isinstance(eps, tuple):
            factored_eps, unfactored_eps = eps
        else:
            factored_eps = unfactored_eps = eps

        defaults = dict(
            lr=lr,
            beta2=beta2,
            factored_eps=factored_eps,
            unfactored_eps=unfactored_eps,
        )
        super().__init__(params, defaults)

    # ----------------------------  core update  ---------------------------- #

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta2: float = group["beta2"]
            factored_eps: float = group["factored_eps"]
            unfactored_eps: float = group["unfactored_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g: Tensor = p.grad  # shortcut

                state = self.state[p]
                d = g.dim()

                # ── 1‑D tensors → unfactored 2‑nd moment ────────────────────
                if d < 2:
                    if "v" not in state:
                        state["v"] = torch.zeros_like(g)

                    v = state["v"]
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    denom = v.sqrt().add_(unfactored_eps)

                # ── ≥2‑D tensors → factored 2‑nd moment ─────────────────────
                else:
                    shape_row = g.shape[:-1]          # all dims except last
                    shape_col = g.shape[-1]           # last dim

                    if "vr" not in state:
                        # Row accumulator ⟨g²⟩ averaged over last dim
                        state["vr"] = torch.zeros(shape_row, dtype=g.dtype, device=g.device)
                        # Column accumulator ⟨g²⟩ averaged over all but last dim
                        state["vc"] = torch.zeros(shape_col, dtype=g.dtype, device=g.device)

                    vr: Tensor = state["vr"]
                    vc: Tensor = state["vc"]

                    # Flatten to 2‑D matrix [R, C] for simplicity
                    g_2d = g.flatten(0, -2)           # R = prod(shape_row)
                    grad_sq = g_2d.pow(2)

                    # Update factored second‑moment estimates
                    vr.mul_(beta2).add_(grad_sq.mean(dim=-1), alpha=1.0 - beta2)
                    vc.mul_(beta2).add_(grad_sq.mean(dim=0),  alpha=1.0 - beta2)

                    # Build the denominator:  √(outer(vr, vc) / mean(vr))
                    #   (Append / unsqueeze so broadcasting works)
                    denom = (
                        (vr.unsqueeze(-1) * vc.unsqueeze(0))
                        .div(vr.mean())
                        .sqrt()
                        .add_(factored_eps)
                    )

                    # reshape denom back to original g shape
                    denom = denom.view_as(g_2d).reshape_as(g)

                update = g / denom
                update = update / (update.norm()+1e-20) * p.data.norm()

                # Parameter update: θ ← θ − lr ⋅ g / denom
                if p.dtype == torch.bfloat16:
                    p.data = add_scaled_bf16b(p.data, update, -lr)
                else:
                    p.add_(update, alpha=-lr)

        return loss

def adafactor_step(p, lr, beta2=0.99, factored_eps=1e-30, unfactored_eps=1e-30, nostround=False):
    g: Tensor = p.grad

    state = p.state
    d = g.dim()

    # ── 1‑D tensors → unfactored 2‑nd moment ────────────────────
    if d < 2:
        if "v" not in state:
            state["v"] = torch.zeros_like(g)

        v = state["v"]
        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

        denom = v.sqrt().add_(unfactored_eps)

    # ── ≥2‑D tensors → factored 2‑nd moment ─────────────────────
    else:
        shape_row = g.shape[:-1]          # all dims except last
        shape_col = g.shape[-1]           # last dim

        if "vr" not in state:
            # Row accumulator ⟨g²⟩ averaged over last dim
            state["vr"] = torch.zeros(shape_row, dtype=g.dtype, device=g.device)
            # Column accumulator ⟨g²⟩ averaged over all but last dim
            state["vc"] = torch.zeros(shape_col, dtype=g.dtype, device=g.device)

        vr: Tensor = state["vr"]
        vc: Tensor = state["vc"]

        # Flatten to 2‑D matrix [R, C] for simplicity
        g_2d = g.flatten(0, -2)           # R = prod(shape_row)
        grad_sq = g_2d.pow(2)

        # Update factored second‑moment estimates
        vr.mul_(beta2).add_(grad_sq.mean(dim=-1), alpha=1.0 - beta2)
        vc.mul_(beta2).add_(grad_sq.mean(dim=0),  alpha=1.0 - beta2)

        # Build the denominator:  √(outer(vr, vc) / mean(vr))
        #   (Append / unsqueeze so broadcasting works)
        denom = (
            (vr.unsqueeze(-1) * vc.unsqueeze(0))
            .div(vr.mean())
            .sqrt()
            .add_(factored_eps)
        )

        # reshape denom back to original g shape
        denom = denom.view_as(g_2d).reshape_as(g)

    update = g / denom
    update = update / (update.norm()+1e-20) * p.data.norm()

    # Parameter update: θ ← θ − lr ⋅ g / denom
    if nostround:
        p.add_(update, alpha=-lr)
    else:
        p.data = add_scaled_bf16(p.data, update, -lr)
