"""Retrieval metrics.

The synthetic dataset has only 3 shapes x 6 colors x 3 backgrounds = 54
possible captions, so many val pairs share the exact same caption. A strict
"the argmax must be index i" Recall@1 counts a retrieved image with an
identical caption as a miss, which under-reports the model. Here a hit means
the retrieved item carries the same caption as the query's ground truth.
"""
from typing import Dict, Sequence

import torch


def _caption_ids(captions: Sequence[str]) -> torch.Tensor:
    lookup: Dict[str, int] = {}
    return torch.tensor([lookup.setdefault(c, len(lookup)) for c in captions])


@torch.no_grad()
def retrieval_recall(image_embeds: torch.Tensor, text_embeds: torch.Tensor,
                     captions: Sequence[str], ks: Sequence[int] = (1, 5)) -> Dict[str, float]:
    """Recall@K in both directions for L2-normalized (N, D) embeddings.

    Row i of image_embeds / text_embeds / captions is the i-th ground-truth pair.
    Returns keys like "t2i_recall_at_1" and "i2t_recall_at_5".
    """
    if not (len(image_embeds) == len(text_embeds) == len(captions)):
        raise ValueError("image_embeds, text_embeds and captions must have the same length")
    ids = _caption_ids(captions).to(image_embeds.device)
    sims = image_embeds @ text_embeds.t()            # (N_img, N_txt)
    n = sims.size(0)
    results: Dict[str, float] = {}
    for k in ks:
        k_eff = min(k, n)
        # image -> text: does any of the top-k texts have the image's caption?
        top_txt = sims.topk(k_eff, dim=1).indices    # (N, k)
        i2t = (ids[top_txt] == ids[:, None]).any(dim=1).float().mean().item()
        # text -> image: does any of the top-k images have the query's caption?
        top_img = sims.t().topk(k_eff, dim=1).indices
        t2i = (ids[top_img] == ids[:, None]).any(dim=1).float().mean().item()
        results[f"i2t_recall_at_{k}"] = i2t
        results[f"t2i_recall_at_{k}"] = t2i
    return results
