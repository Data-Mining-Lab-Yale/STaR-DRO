"""Differentiable annotation objective, main paper sec:stardro_eppc.

Single-process integration. Distributed callers must globally aggregate detached
statistics and use a global weighted-token denominator before using this API.
"""
import torch
import torch.nn.functional as F


def completion_token_loss(logits, labels):
    """Causal CE with prompt/padding targets masked; loss index j predicts j+1."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    valid = labels[:, 1:] != -100
    if not valid.any() or not valid.any(dim=1).all():
        raise ValueError("Each sample must have completion targets")
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                         labels[:, 1:].reshape(-1), ignore_index=-100,
                         reduction="none").reshape(labels.shape[0], -1)
    return ce, valid


def sample_loss(logits, labels, memberships, adversary=None, progress=0.0):
    """Algorithm 1: mean group multipliers and normalized per-example CE.

    Each sample contributes its mean completion CE to its unique memberships,
    with overlap correction inside StarDRO.update. The entire completion gets
    the mean multiplier. Empty semantic memberships keep neutral supervision.
    """
    ce, valid = completion_token_loss(logits, labels)
    if len(memberships) != labels.shape[0]:
        raise ValueError("One membership set required per sample")
    means = ce.sum(1) / valid.sum(1)
    weights = torch.ones_like(means)
    table = None
    if adversary is not None:
        keep = [i for i, members in enumerate(memberships) if members]
        signals = means.detach().cpu().tolist()
        table = adversary.update([signals[i] for i in keep], [memberships[i] for i in keep], progress)
        weights = means.new_tensor([sum(table[g] for g in members) / len(members) if members else 1.0
                                    for members in memberships])
    loss = (means * weights).sum() / weights.sum()
    return loss, {"multipliers": table, "sample_weights": weights.detach(),
                  "token_weights": (weights[:, None] * valid).detach()}


def annotation_loss(logits, labels, annotation_masks, annotation_groups,
                    adversary=None, progress=0.0):
    """Return (weighted completion CE, diagnostics).

    logits: [B,T,V]; labels: [B,T], prompt/padding=-100.
    masks: [B,A,T-1] already aligned with labels[:,1:]; groups: [B,A], pad=-1.
    Annotation masks may overlap at tokenizer boundaries; use their mean weight.
    Passing adversary=None computes ordinary completion CE without state changes.
    Call once per training minibatch, not during evaluation or recomputation.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    b, t = labels.shape
    if annotation_groups.ndim != 2 or annotation_groups.shape[0] != b:
        raise ValueError("Expected annotation_groups [B,A]")
    if annotation_masks.shape != (b, annotation_groups.shape[1], t-1):
        raise ValueError("Expected annotation_masks [B,A,T-1]")
    valid = labels[:, 1:] != -100
    if not valid.any():
        raise ValueError("Batch has no completion targets")
    masks = annotation_masks.bool()
    if (masks & ~valid[:, None, :]).any():
        raise ValueError("Annotation masks cover prompt or padding")
    lengths = masks.sum(-1)
    if ((annotation_groups < 0) & (lengths > 0)).any():
        raise ValueError("Nonempty annotation has no group")
    if ((annotation_groups >= 0) & (lengths == 0)).any():
        raise ValueError("Annotation has no predicted tokens; check truncation")
    token_ce, valid = completion_token_loss(logits, labels)
    weights = torch.ones_like(token_ce)
    table = None
    if adversary is not None:
        keep = annotation_groups >= 0
        means = (token_ce.detach()[:, None, :] * masks).sum(-1) / lengths.clamp_min(1)
        gids = annotation_groups[keep].detach().cpu().tolist()
        signals = means[keep].cpu().tolist()
        table = adversary.update(signals, [[int(g)] for g in gids], progress)
        group_weights = token_ce.new_tensor(table)
        if keep.any():
            ann_weights = torch.ones_like(lengths, dtype=token_ce.dtype)
            ann_weights[keep] = group_weights[annotation_groups[keep]]
            coverage = masks.sum(1)
            routed = (ann_weights[:, :, None] * masks).sum(1) / coverage.clamp_min(1)
            weights = torch.where(coverage > 0, routed, weights)
    weights = weights * valid
    denominator = weights.sum()
    loss = (token_ce * weights).sum() / denominator
    return loss, {"multipliers": table, "weighted_tokens": float(denominator.detach()),
                  "token_weights": weights.detach()}


def structured_loss(logits, batch, granularity="annotation", adversary=None, progress=0.0):
    if granularity == "annotation":
        return annotation_loss(logits, batch["labels"], batch["annotation_masks"],
                               batch["annotation_groups"], adversary, progress)
    if granularity == "sample":
        return sample_loss(logits, batch["labels"], batch["sample_memberships"], adversary, progress)
    raise ValueError("granularity must be annotation or sample")
