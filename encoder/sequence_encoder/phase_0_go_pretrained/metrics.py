import torch
import torch.nn.functional as F


def _pairwise_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity matrix with zero diagonal. Shape: [N, N]."""
    x_norm = F.normalize(x, p=2, dim=1)
    sim = torch.mm(x_norm, x_norm.t())
    sim.fill_diagonal_(0.0)
    return sim


class InterAndIntraClassSimilarity:
    def __init__(self):
        self.intra_class_similarity = torch.tensor(0.0)
        self.inter_class_similarity = torch.tensor(0.0)
        self.count = torch.tensor(0)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Args:
            preds:  Embeddings [B, D]
            target: Class labels [B]
        """
        preds = preds.cpu()
        target = target.cpu()
        cos_sim = _pairwise_cosine_similarity(preds)

        same_label_mask = target.unsqueeze(1) == target.unsqueeze(0)
        upper_triangle_mask = torch.triu(torch.ones_like(same_label_mask), diagonal=1)

        intra_class_sim = cos_sim[same_label_mask & upper_triangle_mask]
        inter_class_sim = cos_sim[(~same_label_mask) & upper_triangle_mask]

        if intra_class_sim.numel() > 0:
            self.intra_class_similarity = self.intra_class_similarity + intra_class_sim.mean()
        if inter_class_sim.numel() > 0:
            self.inter_class_similarity = self.inter_class_similarity + inter_class_sim.mean()
        self.count = self.count + 1

    def compute(self) -> tuple:
        return (
            self.inter_class_similarity / self.count,
            self.intra_class_similarity / self.count,
        )


def compute_batch_mrr(anchor_embed, pos_embed, neg_embed):
    """Compute batch-local MRR using the current mini-batch candidate pool."""
    batch_size = anchor_embed.size(0)
    if batch_size == 0:
        return 0.0

    candidates = torch.cat([pos_embed, neg_embed], dim=0)
    dists = torch.cdist(anchor_embed, candidates, p=2)

    total_mrr = 0.0
    for i in range(batch_size):
        target_idx = i
        sorted_indices = torch.argsort(dists[i], descending=False)
        rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
        total_mrr += 1.0 / rank

    return total_mrr / batch_size


def compute_detailed_metrics(anchor_embed, pos_embed, neg_embed):
    """Compute triplet-level diagnostics for a single batch."""
    metrics = {}

    # ALIGNMENT & UNIFORMITY
    align_loss = (anchor_embed - pos_embed).norm(p=2, dim=1).pow(2).mean()
    metrics["Alignment"] = align_loss.item()

    all_embeds = torch.cat([anchor_embed, pos_embed, neg_embed], dim=0)
    if len(all_embeds) > 1:
        dist_sq = torch.pdist(all_embeds, p=2).pow(2)
        unif_loss = dist_sq.mul(-2).exp().mean().log()
        metrics["Uniformity"] = unif_loss.item()
    else:
        metrics["Uniformity"] = 0.0

    # CLASSIFICATION-LIKE METRIC
    d_pos = F.pairwise_distance(anchor_embed, pos_embed)
    d_neg = F.pairwise_distance(anchor_embed, neg_embed)
    metrics["Simple_Acc"] = (d_pos < d_neg).float().mean().item()
    metrics["MRR"] = compute_batch_mrr(anchor_embed, pos_embed, neg_embed)

    return metrics


def compute_retrieval_mrr(anchor_embed, pos_embed, neg_embed):
    """
    Compute MRR over a full evaluation pool instead of a single mini-batch.

    The candidate pool is formed by all positive and negative embeddings collected
    across the entire validation pass for one ontology.
    """
    batch_size = anchor_embed.size(0)
    if batch_size == 0:
        return 0.0

    candidates = torch.cat([pos_embed, neg_embed], dim=0)
    dists = torch.cdist(anchor_embed, candidates, p=2)

    total_mrr = 0.0
    for i in range(batch_size):
        target_idx = i
        sorted_indices = torch.argsort(dists[i], descending=False)
        rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
        total_mrr += 1.0 / rank

    return total_mrr / batch_size
