import torch
import torch.nn as nn


class TimeWarp(nn.Module):
    """Very light MLP that maps scalar time input to 4096-d embedding.
    Initially initialized to near-zero so it behaves like identity/pass-through.
    """

    def __init__(self, hidden_dim: int = 64, output_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # initialize near zero so effect is initially negligible
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # expects shape (B,1)
        return self.net(t)


class SpaceWarp(nn.Module):
    """Maps normalized (x,y) coords (and optional z) to 4096-dim embedding. Pass-through initialised."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, output_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords shape (B,in_dim)
        return self.net(coords)


class UncertaintyHead(nn.Module):
    """Predicts scalar uncertainty (sigma_u >=0) from latent state."""

    def __init__(self, in_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Softplus()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------
# Goal-Space Warp: Real-NVP style conditional flow ψ(z, g)
# ---------------------------------------------------------------------


class _AffineCoupling(nn.Module):
    """Half-split affine coupling layer with goal-condition."""

    def __init__(self, dim: int, cond_dim: int, hidden: int = 1024):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        # MLP outputs scale and shift for the second half
        self.net = nn.Sequential(
            nn.Linear(dim // 2 + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim // 2 * 2),  # scale & shift
        )

    def forward(self, z: torch.Tensor, g: torch.Tensor, reverse: bool = False):
        """
        Args:
            z: latent tensor (B, D)
            g: goal conditioning (B, cond_dim)
            reverse: if True, applies inverse mapping
        Returns:
            z_out, log_det_jacobian  (B, D),  (B,)
        """
        z1, z2 = torch.chunk(z, 2, dim=-1)
        h = torch.cat([z1, g], dim=-1)
        s_t = self.net(h)
        s, t = torch.chunk(s_t, 2, dim=-1)
        # constrain scale for stability
        s = torch.tanh(s)  # (-1,1) approx => exp(s) in (e^-1, e^1)
        if not reverse:
            y2 = z2 * torch.exp(s) + t
            y = torch.cat([z1, y2], dim=-1)
            log_det = s.sum(dim=-1)
        else:
            y2 = (z2 - t) * torch.exp(-s)
            y = torch.cat([z1, y2], dim=-1)
            log_det = -s.sum(dim=-1)
        return y, log_det


class GoalWarp(nn.Module):
    """Real-NVP style conditional flow that warps latent state by goal vector."""

    def __init__(self, latent_dim: int = 4096, goal_dim: int = 16, num_flows: int = 4, hidden: int = 1024):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even for coupling split"
        self.latent_dim = latent_dim
        self.goal_dim = goal_dim
        self.flows = nn.ModuleList([
            _AffineCoupling(latent_dim, goal_dim, hidden) for _ in range(num_flows)
        ])
        # simple fixed permutation indices (alternating)
        perm = torch.arange(latent_dim - 1, -1, -1)  # reverse
        self.register_buffer("_perm", perm, persistent=False)

    def _permute(self, z: torch.Tensor, reverse: bool = False):
        return z[:, self._perm] if not reverse else z[:, self._perm]

    def forward(self, z: torch.Tensor, g: torch.Tensor, reverse: bool = False, return_logdet: bool = False):
        """Applies the forward (warp) or inverse (counter-warp) mapping.

        Returns warped tensor and optional log|det J|.
        """
        log_det_total = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        if not reverse:
            for flow in self.flows:
                z, ld = flow(z, g, reverse=False)
                log_det_total += ld
                z = self._permute(z)
        else:
            for flow in reversed(self.flows):
                z = self._permute(z, reverse=True)
                z, ld = flow(z, g, reverse=True)
                log_det_total += ld
        if return_logdet:
            return z, log_det_total
        return z 