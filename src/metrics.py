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


@torch.no_grad()
def grouped_recall(image_embeds: torch.Tensor, image_ids: Sequence, text_embeds: torch.Tensor,
                   text_image_ids: Sequence, ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """Standard Flickr30k / COCO retrieval protocol (each image has several captions).

    image -> text: an image query is correct if ANY of its own captions is in the top-K texts.
    text -> image: a caption query is correct if ITS image is in the top-K images.
    "rsum" (sum of all six R@K in %) is the usual single number for model selection.
    """
    if len(image_embeds) != len(image_ids) or len(text_embeds) != len(text_image_ids):
        raise ValueError("embeddings and ids must have the same length")
    lookup: Dict = {}
    img_gid = torch.tensor([lookup.setdefault(i, len(lookup)) for i in image_ids])
    if len(lookup) != len(image_ids):
        raise ValueError("image_ids must be unique")
    try:
        txt_gid = torch.tensor([lookup[i] for i in text_image_ids])
    except KeyError as e:
        raise ValueError(f"caption refers to an image not in the gallery: {e}") from None
    img_gid, txt_gid = img_gid.to(image_embeds.device), txt_gid.to(image_embeds.device)

    sims = image_embeds.float() @ text_embeds.float().t()      # (N_img, N_txt)
    results: Dict[str, float] = {}
    for k in ks:
        top_txt = sims.topk(min(k, sims.size(1)), dim=1).indices
        results[f"i2t_recall_at_{k}"] = (txt_gid[top_txt] == img_gid[:, None]).any(1).float().mean().item()
        top_img = sims.t().topk(min(k, sims.size(0)), dim=1).indices
        results[f"t2i_recall_at_{k}"] = (img_gid[top_img] == txt_gid[:, None]).any(1).float().mean().item()
    results["rsum"] = 100 * sum(v for v in results.values())
    return results
