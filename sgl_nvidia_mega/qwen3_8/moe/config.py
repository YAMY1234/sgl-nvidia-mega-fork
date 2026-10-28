# SPDX-License-Identifier: Apache-2.0
"""HF-derived FN MoE dimensions and explicit current-engine disable reasons."""
from dataclasses import dataclass


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


@dataclass(frozen=True)
class Shape:
    hidden: int
    experts: int
    top_k: int
    intermediate: int
    tp_size: int
    max_rows: int

    @classmethod
    def from_config(cls, hf_config, tp_size: int, max_rows: int):
        text = _get(hf_config, 'text_config', hf_config)
        return cls(int(_get(text, 'hidden_size', 0)), int(_get(text, 'num_experts', 0)),
                   int(_get(text, 'num_experts_per_tok', 0)),
                   int(_get(text, 'moe_intermediate_size', 0)), int(tp_size), int(max_rows))

    @property
    def local_intermediate(self):
        return self.intermediate // self.tp_size if self.tp_size > 0 else 0

    @property
    def storage_intermediate(self):
        # ModelOpt gated loading rounds a TP shard to64 rows. It slices the
        # checkpoint at storage192*rank, not logical160*rank +32 tail padding.
        return ((self.local_intermediate + 63) // 64) * 64

    @property
    def mkm_capacity(self):
        return 7  # compiled small-M engine capacity; not tokens_per_request

    def compile_options(self):
        return (f'-DFN_HIDDEN={self.hidden}', f'-DFN_EXPERTS={self.experts}',
                f'-DFN_TOP_K={self.top_k}', f'-DFN_STORAGE_I={self.storage_intermediate}')


def config_disable_reasons(hf_config, tp_size: int, max_rows: int, slot='mkm'):
    shape = Shape.from_config(hf_config, tp_size, max_rows)
    reasons = []
    if shape.tp_size <= 0 or shape.intermediate <= 0 or shape.intermediate % max(shape.tp_size, 1):
        reasons.append('MoE intermediate must divide positive tensor-parallel size')
    if (shape.hidden, shape.experts, shape.top_k) != (2560, 512, 10):
        reasons.append(f'FN MoE engine supports H2560/E512/top10; got H{shape.hidden}/E{shape.experts}/top{shape.top_k}')
    if not 1 <= shape.max_rows <= (shape.mkm_capacity if slot=='mkm' else 16):
        reasons.append(f'{slot}: requested{shape.max_rows} rows exceeds current engine capacity')
    if slot=='mkm' and (shape.local_intermediate,shape.storage_intermediate)!=(160,192):
        reasons.append(f'FN SF MKM supports logical I160/storage I192; got logical{shape.local_intermediate}/storage{shape.storage_intermediate}')
    if slot=='route_shared':
        text = _get(hf_config,'text_config',hf_config)
        shared_i = int(_get(text,'shared_expert_intermediate_size',0))
        if shared_i <= 0 or shared_i % max(shape.tp_size,1):
            reasons.append('shared intermediate must divide positive TP size')
        if shape.max_rows > 8:
            reasons.append('BF16 shared gate/up engine currently covers at most8 rows')
    return reasons
