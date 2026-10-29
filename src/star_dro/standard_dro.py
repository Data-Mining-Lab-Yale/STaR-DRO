"""Notebook144 Standard DRO: exponential q update and annotation-count correction."""
import math


class StandardDRO:
    def __init__(self, num_groups, eta=0.1):
        if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups < 1:
            raise ValueError("num_groups must be a positive integer")
        if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta <= 0:
            raise ValueError("standard_dro.eta must be finite and positive")
        self.num_groups, self.eta = num_groups, eta
        self.logq = [-math.log(num_groups)] * num_groups
        self.q = [1 / num_groups] * num_groups
        self.ema = [None] * num_groups  # Standard DRO has no EMA.
        self.steps, self.last_update = 0, None

    def update(self, losses, memberships, progress=0.0):
        if len(losses) != len(memberships):
            raise ValueError("losses and memberships must align")
        sums, counts = [0.0] * self.num_groups, [0] * self.num_groups
        for loss, members in zip(losses, memberships):
            if not math.isfinite(loss) or loss < 0 or len(members) != 1:
                raise ValueError("Standard DRO requires nonnegative annotation losses with one group per annotation")
            group = members[0]
            if isinstance(group, bool) or not isinstance(group, int) or not 0 <= group < self.num_groups:
                raise ValueError("invalid Standard DRO group index")
            sums[group] += loss
            counts[group] += 1
        present = [g for g, count in enumerate(counts) if count]
        if not present:
            return [1.0] * self.num_groups
        means = [sums[g] / counts[g] if counts[g] else 0.0 for g in range(self.num_groups)]
        logq = [self.logq[g] + self.eta * means[g] for g in range(self.num_groups)]
        # Equivalent to q <- softmax(log(q) + eta*group_loss), retaining the
        # log-state so a tiny absent-group mass can recover without log(0).
        top = max(logq)
        lognormalizer = top + math.log(sum(math.exp(value - top) for value in logq))
        self.logq = [value - lognormalizer for value in logq]
        self.q = [math.exp(value) for value in self.logq]
        # Notebook144: m_g = q_g / p_hat_g / sum(q_present). Compute the
        # conditional mass in log space to remain finite for very small q.
        top_present = max(self.logq[g] for g in present)
        mass = sum(math.exp(self.logq[g] - top_present) for g in present)
        total = sum(counts)
        multipliers = [1.0] * self.num_groups
        for g in present:
            conditional = math.exp(self.logq[g] - top_present) / mass
            multipliers[g] = conditional / (counts[g] / total)
        self.steps += 1
        self.last_update = {"objective": "standard_dro", "eta": self.eta,
                            "group_losses": means, "annotation_counts": counts,
                            "q": list(self.q), "multipliers": multipliers}
        return multipliers

    def state_dict(self):
        return {"objective": "standard_dro", "num_groups": self.num_groups, "eta": self.eta,
                "logq": list(self.logq), "q": list(self.q), "steps": self.steps,
                "last_update": self.last_update}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("objective") != "standard_dro":
            raise ValueError("not a Standard DRO checkpoint")
        result = cls(state["num_groups"], state["eta"])
        logq = state["logq"]
        if len(logq) != result.num_groups or not all(math.isfinite(x) for x in logq):
            raise ValueError("invalid Standard DRO logq checkpoint")
        if abs(sum(math.exp(x) for x in logq) - 1) > 1e-8:
            raise ValueError("Standard DRO checkpoint q must sum to one")
        if isinstance(state["steps"], bool) or not isinstance(state["steps"], int) or state["steps"] < 0:
            raise ValueError("invalid Standard DRO step count")
        result.logq = list(logq)
        result.q = [math.exp(x) for x in logq]
        result.steps, result.last_update = state["steps"], state.get("last_update")
        return result
