import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


def _build_backbone(model_name: str, pretrained: bool = True, config=None) -> nn.Module:
    """Build a Hugging Face backbone.

    pretrained=True  -> download/load the pretrained weights (training).
    pretrained=False -> build the architecture from its config only; the
                        weights are expected to come from our own checkpoint
                        (serving), so the backbone weights are not loaded twice.
    config           -> an explicit transformers config object (used by the
                        unit tests to build tiny models fully offline).
    """
    if config is not None:
        return AutoModel.from_config(config)
    if pretrained:
        return AutoModel.from_pretrained(model_name)
    return AutoModel.from_config(AutoConfig.from_pretrained(model_name))


class VisionEncoder(nn.Module):
    """Wraps any Hugging Face vision foundation model (ViT-style backbone
    that exposes `last_hidden_state`, e.g. DINOv2, ViT, BEiT)."""

    def __init__(self, model_name: str, freeze: bool = True,
                 pretrained: bool = True, config=None):
        super().__init__()
        self.backbone = _build_backbone(model_name, pretrained, config)
        self.hidden_size = self.backbone.config.hidden_size
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        # model.train() would otherwise flip a frozen backbone back into train
        # mode and re-enable its dropout, adding noise to "frozen" features.
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor, pooled: bool = True) -> torch.Tensor:
        with torch.no_grad() if self.freeze else torch.enable_grad():
            out = self.backbone(pixel_values=pixel_values)
        # out.last_hidden_state: (B, num_patches + 1, hidden). Index 0 is the CLS token.
        return out.last_hidden_state[:, 0] if pooled else out.last_hidden_state


class TextEncoder(nn.Module):
    """Wraps any Hugging Face text encoder (BERT/DistilBERT/MiniLM-style)."""

    def __init__(self, model_name: str, freeze: bool = True,
                 pretrained: bool = True, config=None):
        super().__init__()
        self.backbone = _build_backbone(model_name, pretrained, config)
        self.hidden_size = self.backbone.config.hidden_size
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                pooled: bool = True) -> torch.Tensor:
        with torch.no_grad() if self.freeze else torch.enable_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if not pooled:
            return out.last_hidden_state
        # Mean-pool token embeddings, ignoring padding.
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


class ProjectionHead(nn.Module):
    """Small trainable MLP that maps a frozen encoder's feature space into
    the shared fusion embedding space. This is the only part of the model
    that gets trained by default -> "light training for the fusion"."""

    def __init__(self, in_dim: int, out_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(p_drop),
        )
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(x))


class FusionCLIPModel(nn.Module):
    """CLIP-style dual encoder: frozen HF vision + text foundation models,
    each followed by a trainable projection head into a shared embedding
    space, aligned with a symmetric InfoNCE (contrastive) loss."""

    def __init__(self, vision_model_name: str, text_model_name: str,
                 embed_dim: int = 256, freeze_vision: bool = True,
                 freeze_text: bool = True, pretrained: bool = True,
                 vision_config=None, text_config=None):
        super().__init__()
        self.vision_encoder = VisionEncoder(vision_model_name, freeze=freeze_vision,
                                            pretrained=pretrained, config=vision_config)
        self.text_encoder = TextEncoder(text_model_name, freeze=freeze_text,
                                        pretrained=pretrained, config=text_config)
        self.vision_proj = ProjectionHead(self.vision_encoder.hidden_size, embed_dim)
        self.text_proj = ProjectionHead(self.text_encoder.hidden_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(pixel_values, pooled=True)
        return F.normalize(self.vision_proj(feats), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        feats = self.text_encoder(input_ids, attention_mask, pooled=True)
        return F.normalize(self.text_proj(feats), dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        scale = self.logit_scale.exp()
        logits_per_image = scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def clip_contrastive_loss(logits_per_image: torch.Tensor,
                           logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = logits_per_image.size(0)
    targets = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) / 2


class CrossAttentionFusion(nn.Module):
    """Optional, more tightly-coupled fusion head for tasks that need a
    single joint representation instead of two aligned embeddings (e.g.
    VQA, image-text matching classification). Text tokens attend over the
    full image patch-token sequence. Not wired into the default training
    loop -- use in place of the projection heads for those tasks by calling
    the encoders with `pooled=False` to get full token sequences."""

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.pool = nn.Linear(dim, dim)

    def forward(self, text_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        fused = self.decoder(tgt=text_tokens, memory=image_tokens)
        pooled = fused.mean(dim=1)
        return torch.tanh(self.pool(pooled))
