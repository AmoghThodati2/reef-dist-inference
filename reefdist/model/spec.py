"""ModelSpec: the shapes of a GPT Neo model, read from its Hugging Face config."""
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    d_model: int
    d_ff: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_positions: int
    attention_types: Tuple[str, ...]  # one of "global" / "local" per layer
    window_size: int  # local layers: causal sliding window (keys i-window < j <= i)
    layer_norm_eps: float
    activation: str

    @classmethod
    def from_hf_config(cls, cfg, name: str) -> "ModelSpec":
        if cfg.model_type != "gpt_neo":
            raise ValueError(f"{name}: only GPT Neo models are supported (got model_type={cfg.model_type})")
        d_ff = cfg.intermediate_size if cfg.intermediate_size is not None else 4 * cfg.hidden_size
        return cls(
            name=name,
            num_layers=cfg.num_layers,
            d_model=cfg.hidden_size,
            d_ff=d_ff,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_heads,  # GPT Neo has no grouped-query attention
            head_dim=cfg.hidden_size // cfg.num_heads,
            vocab_size=cfg.vocab_size,
            max_positions=cfg.max_position_embeddings,
            attention_types=tuple(cfg.attention_layers),
            window_size=cfg.window_size,
            layer_norm_eps=cfg.layer_norm_epsilon,
            activation=cfg.activation_function,
        )

    @classmethod
    def from_pretrained(cls, name: str) -> "ModelSpec":
        return _spec_from_pretrained(name)

    def window(self, layer: int) -> Optional[int]:
        """Sliding-window size for a local layer, None for a global layer."""
        return self.window_size if self.attention_types[layer] == "local" else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["attention_types"] = list(self.attention_types)
        return d


@lru_cache(maxsize=None)
def _spec_from_pretrained(name: str) -> ModelSpec:
    from transformers import AutoConfig

    return ModelSpec.from_hf_config(AutoConfig.from_pretrained(name), name)
