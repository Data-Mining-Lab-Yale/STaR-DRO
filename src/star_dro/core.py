"""Algorithm 1 and Steps 1--7 of STaR-DRO.

The adversary uses Python double precision and detached loss statistics.
Its O(G) state is independent of the model and can be serialized as JSON.
"""
from dataclasses import asdict, dataclass
import math


def schedule(start: float, end: float, progress: float) -> float:
    """Cosine interpolation over the entire continuous training run."""
    p = min(1.0, max(0.0, progress))
    return end + (start - end) * (1 + math.cos(math.pi * p)) / 2


def backmap(u, alpha: float):
    """Solve sum([u-lambda]_+**(1/(alpha-1)))=1 by bisection.

    Subtracting max(u) preserves the answer and bounds all powers by one.
    Negative dual coordinates are supported.
    """
    u = [float(x) for x in u]
    if not u or not math.isfinite(alpha) or alpha <= 1:
        raise ValueError("Nonempty dual vector and finite alpha > 1 required")
    if not all(math.isfinite(x) for x in u):
        raise ValueError("Dual coordinates must be finite")
    top = max(u)
    v = [x - top for x in u]
    lo, hi = -1.0, 0.0
    power = 1 / (alpha - 1)
    for _ in range(100):
        mid = (lo + hi) / 2
        mass = sum(max(x - mid, 0.0) ** power for x in v)
        if mass > 1:
            lo = mid
        else:
            hi = mid
    threshold = (lo + hi) / 2
    q = [max(x - threshold, 0.0) ** power for x in v]
    total = sum(q)
    return [x / total for x in q]


@dataclass(frozen=True)
class Config:
    """STaR-DRO hyperparameters; see docs/configuration.md."""
    num_groups: int
    alpha: float = 1.2
    eta: float = 0.0006
    rho_start: float = 0.3
    rho_end: float = 0.03
    upper: float = 15.0
    gamma: float = 0.75
    hardness_scale: str = "count"

    def __post_init__(self):
        if isinstance(self.num_groups, bool) or not isinstance(self.num_groups, int) or self.num_groups < 1:
            raise ValueError("num_groups must be a positive integer")
        vals = [self.alpha, self.eta, self.rho_start, self.rho_end, self.upper, self.gamma]
        if not all(math.isfinite(x) for x in vals):
            raise ValueError("Configuration values must be finite")
        if self.alpha <= 1 or self.eta <= 0 or self.upper <= 1 or self.gamma <= 0:
            raise ValueError("Require alpha>1, eta>0, upper>1, gamma>0")
        if not 0 < self.rho_start <= 1 or not 0 < self.rho_end <= 1:
            raise ValueError("EMA coefficients must be in (0,1]")
        if not isinstance(self.hardness_scale, str) or self.hardness_scale not in {"uniform", "count"}:
            raise ValueError("hardness_scale must be uniform or count")


class StarDRO:
    """Persistent adversary. Never reset EMA at epoch boundaries.

    `update` accepts detached nonnegative losses and unique memberships for
    each observation. An annotation is one observation with one membership.
    For sample attribution, use the sample's mean completion CE and group set.
    """
    def __init__(self, config: Config):
        self.config = config
        self.q = [1 / config.num_groups] * config.num_groups
        self.ema = [None] * config.num_groups
        self.steps = 0
        # Diagnostics only; persistent algorithm state is q and ema.
        self.last_update = None

    def update(self, losses, memberships, progress=0.0):
        g = self.config.num_groups
        sums, mass, counts = [0.0] * g, [0.0] * g, [0] * g
        losses, memberships = list(losses), list(memberships)
        if len(losses) != len(memberships):
            raise ValueError("One membership set required per loss")
        for loss, members in zip(losses, memberships):
            loss = float(loss)
            members = list(members)
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("Losses must be finite and nonnegative")
            if not members or len(set(members)) != len(members):
                raise ValueError("Memberships must be nonempty and unique")
            if any(isinstance(j, bool) or not isinstance(j, int) or not 0 <= j < g for j in members):
                raise ValueError("Group index outside the fixed inventory")
            for j in members:
                sums[j] += loss / len(members)
                mass[j] += 1 / len(members)
                counts[j] += 1
        present = [j for j in range(g) if mass[j] > 0]
        if not present:
            self.last_update = {"present_groups": [], "scale": 0.0,
                                "ascent_signal": [0.0] * g, "multipliers": [1.0] * g}
            return [1.0] * g
        # Component 1: stateful EMA (paper Step 2, eq:ema_loss).
        # Keep the same adversary instance across epochs.
        rho = schedule(self.config.rho_start, self.config.rho_end, progress)
        for j in present:
            avg = sums[j] / mass[j]
            self.ema[j] = avg if self.ema[j] is None else (1 - rho) * self.ema[j] + rho * avg
        # Component 2: rescaled ascent signal (Step 3, eq:ascent_signal).
        # Notebook157 uses a uniform mean; the paper/reference uses counts.
        if self.config.hardness_scale == "uniform":
            scale = sum(self.ema[j] for j in present) / len(present)
        else:
            scale = sum(counts[j] * self.ema[j] for j in present) / sum(counts)
        signal = [0.0] * g
        for j in present:
            signal[j] = self.ema[j] / max(scale, 1e-6)
        power = self.config.alpha - 1
        dual = [q ** power + power * self.config.eta * a for q, a in zip(self.q, signal)]
        self.q = backmap(dual, self.config.alpha)
        self.steps += 1
        multipliers = [1.0] * g
        # Component 3: bounded excess-only multiplier (Step 6, eq:multiplier).
        if g > 1:
            for j in present:
                excess = min(1.0, max(0.0, (g * self.q[j] - 1) / (g - 1)))
                multipliers[j] = 1 + (self.config.upper - 1) * excess ** self.config.gamma
        self.last_update = {"present_groups": present, "rho": rho, "scale": scale,
                            "hardness_scale": self.config.hardness_scale,
                            "ascent_signal": signal, "multipliers": multipliers.copy(),
                            "entropy": -sum(q * math.log(q) for q in self.q if q > 0),
                            "effective_groups": math.exp(-sum(q * math.log(q) for q in self.q if q > 0)),
                            "support_size": sum(q > 1e-12 for q in self.q)}
        return multipliers

    def state_dict(self):
        return {"config": asdict(self.config), "q": self.q.copy(),
                "ema": self.ema.copy(), "steps": self.steps}

    @classmethod
    def from_state_dict(cls, state):
        result = cls(Config(**state["config"]))
        q, ema = list(state["q"]), list(state["ema"])
        if len(q) != result.config.num_groups or len(ema) != len(q):
            raise ValueError("Invalid checkpoint shape")
        if not all(math.isfinite(x) and x >= 0 for x in q) or not math.isclose(sum(q), 1, abs_tol=1e-9):
            raise ValueError("Invalid simplex checkpoint")
        if any(x is not None and (not math.isfinite(x) or x < 0) for x in ema):
            raise ValueError("Invalid EMA checkpoint")
        if not isinstance(state["steps"], int) or state["steps"] < 0:
            raise ValueError("Invalid step count")
        result.q, result.ema, result.steps = q, ema, state["steps"]
        return result
