import torch

from conftest import EMBED_DIM
from src.config import Config, config_from_checkpoint
from src.model import FusionCLIPModel, ProjectionHead, clip_contrastive_loss


def _batch(model, n=3):
    pixel_values = torch.randn(n, 3, 28, 28)
    input_ids = torch.randint(5, 20, (n, 6))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 4:] = 0  # one padded sequence
    return pixel_values, input_ids, attention_mask


def test_embeddings_have_shared_dim_and_unit_norm(tiny_model):
    pixel_values, input_ids, attention_mask = _batch(tiny_model)
    img = tiny_model.encode_image(pixel_values)
    txt = tiny_model.encode_text(input_ids, attention_mask)
    assert img.shape == txt.shape == (3, EMBED_DIM)
    torch.testing.assert_close(img.norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(txt.norm(dim=-1), torch.ones(3))


def test_projection_head_shape():
    assert ProjectionHead(32, 8)(torch.randn(4, 32)).shape == (4, 8)


def test_contrastive_loss_rewards_matching_pairs():
    n = 4
    perfect = torch.eye(n) * 50
    random_logits = torch.randn(n, n)
    assert clip_contrastive_loss(perfect, perfect.t()) < 1e-3
    random_loss = clip_contrastive_loss(random_logits, random_logits.t())
    assert clip_contrastive_loss(perfect, perfect.t()) < random_loss


def test_loss_is_symmetric_average():
    logits = torch.randn(5, 5)
    a = clip_contrastive_loss(logits, logits.t())
    b = clip_contrastive_loss(logits.t(), logits)
    torch.testing.assert_close(a, b)


def test_frozen_backbones_stay_in_eval_mode(tiny_backbones):
    model = FusionCLIPModel(*tiny_backbones, embed_dim=EMBED_DIM)
    model.train()
    # Regression test: model.train() used to flip the frozen backbones back to train
    # mode, re-enabling their dropout while their weights were supposed to be fixed.
    assert not model.vision_encoder.backbone.training
    assert not model.text_encoder.backbone.training
    assert model.vision_proj.training and model.text_proj.training
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(not n.startswith(("vision_encoder.backbone", "text_encoder.backbone"))
                             for n in trainable)


def test_serving_build_matches_training_model(tiny_backbones, tiny_model):
    """pretrained=False (config only) + our checkpoint must reproduce the trained model
    exactly -- otherwise serving would silently differ from what was evaluated."""
    served = FusionCLIPModel(*tiny_backbones, embed_dim=EMBED_DIM, pretrained=False)
    served.load_state_dict(tiny_model.state_dict())
    served.eval()
    pixel_values, input_ids, attention_mask = _batch(tiny_model)
    with torch.no_grad():
        torch.testing.assert_close(served.encode_image(pixel_values), tiny_model.encode_image(pixel_values))
        torch.testing.assert_close(served.encode_text(input_ids, attention_mask),
                                   tiny_model.encode_text(input_ids, attention_mask))


def test_config_from_checkpoint_ignores_training_device_and_unknown_keys():
    ckpt = {"config": {**vars(Config(embed_dim=64)), "device": "mps", "removed_field": 1}}
    cfg = config_from_checkpoint(ckpt, device="cpu")
    assert cfg.device == "cpu" and cfg.embed_dim == 64
    assert config_from_checkpoint(ckpt).device != "mps" or torch.backends.mps.is_available()
