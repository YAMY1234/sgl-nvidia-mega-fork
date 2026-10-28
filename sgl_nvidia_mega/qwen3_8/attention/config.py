"""FN attention dimensions and launch support, without importing CUDA packages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimensions:
    hidden: int
    query_heads: int
    kv_heads: int
    head_dim: int
    budget: int
    ratio: int
    index_heads: int
    index_dim: int

    @classmethod
    def from_config(cls, hf_config):
        text = getattr(hf_config, "text_config", hf_config)
        return cls(*(getattr(text, key, None) for key in (
            "hidden_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "indexer_budget", "indexer_compress_ratio",
            "indexer_n_heads", "indexer_head_dim",
        )))

    def local_heads(self, tp_size):
        return self.query_heads // tp_size, max(1, self.kv_heads // tp_size)


def config_disable_reasons(hf_config, tp_size, tokens_per_request):
    dims = Dimensions.from_config(hf_config)
    reasons = []
    for name, want in (
        ("hidden", 2560), ("query_heads", 24), ("kv_heads", 2),
        ("head_dim", 256), ("budget", 2048), ("ratio", 4),
        ("index_heads", 4), ("index_dim", 128),
    ):
        value = getattr(dims, name)
        if value != want:
            reasons.append(f"FN attention {name} {value!r}; supported {want}")
    text = getattr(hf_config, "text_config", hf_config)
    if getattr(text, "indexer_kv_heads", None) != 1:
        reasons.append("FN attention requires one indexer KV head")
    if tp_size != 4:
        reasons.append(f"FN attention TP {tp_size}; supported TP4")
    if tokens_per_request != 4:
        reasons.append(f"FN attention rows/request {tokens_per_request}; supported 4")
    return reasons
