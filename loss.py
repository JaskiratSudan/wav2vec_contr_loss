# loss.py
import torch
import torch.nn as nn

class SupConBinaryLoss(nn.Module):
    """
    Supervised Contrastive (binary) with alpha-blend:
      loss = (1 - alpha) * SupCon_full + alpha * SupCon_mined
    where SupCon_mined uses the top-K hardest negatives per anchor.
    """
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.tau = temperature

    def _supcon_full(self, sim_row: torch.Tensor, pos_mask_row: torch.Tensor) -> torch.Tensor:
        """
        One-anchor vanilla SupCon over all non-self samples.
        sim_row: (B,) similarity to everyone (self has been -inf).
        pos_mask_row: (B,) bool for positives (self already False).
        """
        # logits for all non-self
        logits_all = sim_row / self.tau                               # (B,)
        # if no positives for this anchor, skip
        if not pos_mask_row.any():
            return None
        pos_logits = logits_all[pos_mask_row]                         # (P,)
        log_prob = pos_logits - torch.logsumexp(logits_all, dim=0)    # (P,)
        return -log_prob.mean()

    def _supcon_mined_topk(self, sim_row: torch.Tensor, pos_mask_row: torch.Tensor,
                           neg_mask_row: torch.Tensor, topk_neg: int) -> torch.Tensor:
        """
        One-anchor mined SupCon:
          - positives: all positives
          - negatives: top-K most similar negatives
        """
        if not pos_mask_row.any() or not neg_mask_row.any():
            return None

        pos_sims = sim_row[pos_mask_row]                              # (P,)
        neg_sims = sim_row[neg_mask_row]                              # (N,)
        # top-K hardest negatives
        k = min(topk_neg, neg_sims.numel())
        if k < 1:
            return None
        neg_topk, _ = torch.sort(neg_sims, descending=True)
        neg_topk = neg_topk[:k]                                       # (k,)

        denom = torch.cat([pos_sims, neg_topk], dim=0)                # (P+k,)
        logits = denom / self.tau
        log_probs = logits - torch.logsumexp(logits, dim=0)
        return -log_probs[:pos_sims.numel()].mean()

    def forward(self,
                z: torch.Tensor,        # (B, D) L2-normalized embeddings
                labels: torch.Tensor,   # (B,) in {0,1}
                topk_neg: int = 32,
                alpha: float = 0.0) -> torch.Tensor:

        device = z.device
        B = z.size(0)
        # cosine sim; mask self
        sim = z @ z.t()                                              # (B,B)
        eye = torch.eye(B, device=device, dtype=torch.bool)
        sim = sim.masked_fill(eye, float("-inf"))

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & (~eye)                   # (B,B)
        neg_mask = (~pos_mask) & (~eye)

        losses_full = []
        losses_mined = []
        for i in range(B):
            lf = self._supcon_full(sim[i], pos_mask[i])
            lm = self._supcon_mined_topk(sim[i], pos_mask[i], neg_mask[i], topk_neg)
            if lf is not None:
                losses_full.append(lf)
            if lm is not None:
                losses_mined.append(lm)

        if len(losses_full) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss_full = torch.stack(losses_full).mean()
        if len(losses_mined) == 0:
            loss_mined = loss_full
        else:
            loss_mined = torch.stack(losses_mined).mean()

        # Blend difficulty by alpha
        return (1.0 - alpha) * loss_full + alpha * loss_mined

class SupConMultiClassLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al.) for multi-class labels.
    Treats each distinct label ID (0..C-1) as its own class.
    For each anchor:
      - positives: same-class samples in the batch (excluding self)
      - negatives: all other samples in the batch
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self,
                z: torch.Tensor,        # (B, D) L2-normalized embeddings
                labels: torch.Tensor    # (B,) long, values in {0..C-1}
                ) -> torch.Tensor:
        device = z.device
        B = z.size(0)
        assert labels.dim() == 1 and labels.size(0) == B, "labels must be shape (B,)"

        # Cosine sim (since z normalized, dot product)
        sim = torch.matmul(z, z.t()) / self.tau  # (B, B)

        # Mask out self-comparisons
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, float('-inf'))

        labels = labels.view(-1, 1)  # (B, 1)
        eq = (labels == labels.t())  # (B, B)
        pos_mask = eq & (~self_mask)  # same class, not self

        losses = []

        for i in range(B):
            pos_idx = torch.nonzero(pos_mask[i], as_tuple=False).squeeze(-1)
            if pos_idx.numel() == 0:
                # No positive of same class in batch -> skip anchor
                continue

            # Positives for anchor i
            pos_logits = sim[i, pos_idx]  # (P,)

            # Denominator: all non-self logits (both pos+neg)
            all_logits = sim[i, ~self_mask[i]]  # (B-1,)

            # log p(pos | all)
            log_prob = pos_logits - torch.logsumexp(all_logits, dim=0)  # (P,)
            loss_i = -log_prob.mean()
            losses.append(loss_i)

        if len(losses) == 0:
            # Degenerate tiny-batch case
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses, dim=0).mean()
