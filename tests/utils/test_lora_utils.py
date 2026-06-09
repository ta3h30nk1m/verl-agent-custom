import torch.nn as nn

from verl.utils.lora_utils import find_language_layer_classes, resolve_lora_target_modules


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(4, 4),
                "k_proj": nn.Linear(4, 4),
                "v_proj": nn.Linear(4, 4),
                "o_proj": nn.Linear(4, 4),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(4, 4),
                "up_proj": nn.Linear(4, 4),
                "down_proj": nn.Linear(4, 4),
            }
        )


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)


class FakeMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.layers = nn.ModuleList([EncoderBlock()])
        self.audio_encoder = nn.Module()
        self.audio_encoder.layers = nn.ModuleList([EncoderBlock()])
        self.multi_modal_projector = nn.Linear(4, 4)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = nn.Embedding(8, 4)
        self.language_model.model.layers = nn.ModuleList([AttentionBlock(), AttentionBlock()])
        self.lm_head = nn.Linear(4, 8)


def test_resolve_lora_target_modules_llm_attention_excludes_encoders_and_heads():
    model = FakeMultimodalModel()

    targets = resolve_lora_target_modules(model, "all-linear", "llm_attention")

    assert len(targets) == 8
    assert all(".self_attn." in target for target in targets)
    assert all("language_model.model.layers" in target for target in targets)
    assert not any("vision" in target or "audio" in target for target in targets)
    assert not any("multi_modal_projector" in target or "lm_head" in target for target in targets)


def test_resolve_lora_target_modules_llm_all_linear_keeps_mlp_inside_language_layers():
    model = FakeMultimodalModel()

    targets = resolve_lora_target_modules(model, "all-linear", "llm")

    assert len(targets) == 14
    assert any(target.endswith(".mlp.gate_proj") for target in targets)
    assert any(target.endswith(".self_attn.q_proj") for target in targets)
    assert not any("vision" in target or "audio" in target for target in targets)


def test_resolve_lora_target_modules_filters_explicit_names_inside_language_layers():
    model = FakeMultimodalModel()

    targets = resolve_lora_target_modules(model, ["q_proj", "v_proj"], "llm")

    assert len(targets) == 4
    assert all(target.endswith((".q_proj", ".v_proj")) for target in targets)


def test_find_language_layer_classes_ignores_multimodal_encoder_layers():
    model = FakeMultimodalModel()

    classes = find_language_layer_classes(model)

    assert classes == {AttentionBlock}
