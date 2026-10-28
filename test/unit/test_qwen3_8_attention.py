"""CPU contract tests for FN dimensions and the generic ForwardView route."""

from types import SimpleNamespace as NS
from unittest import TestCase

from sgl_nvidia_mega.qwen3_8.attention.config import Dimensions, config_disable_reasons
from sgl_nvidia_mega.qwen3_8.attention.qsa_core import metadata_reasons


def hf_config(**changes):
    fields = dict(hidden_size=2560, num_attention_heads=24, num_key_value_heads=2,
                  head_dim=256, indexer_budget=2048, indexer_compress_ratio=4,
                  indexer_n_heads=4, indexer_head_dim=128, indexer_kv_heads=1)
    fields.update(changes)
    return NS(text_config=NS(**fields))


class TestDimensions(TestCase):
    def test_replication_uses_one_local_kv_head_at_tp4(self):
        dims = Dimensions.from_config(hf_config())
        self.assertEqual(dims.local_heads(4), (6, 1))
        self.assertEqual(config_disable_reasons(hf_config(), 4, 4), [])

    def test_config_changes_are_named_before_any_kernel_launch(self):
        why = config_disable_reasons(hf_config(hidden_size=4096, indexer_budget=1024), 8, 7)
        self.assertTrue(any("hidden 4096" in x for x in why))
        self.assertTrue(any("budget 1024" in x for x in why))
        self.assertTrue(any("TP 8" in x for x in why))
        self.assertTrue(any("rows/request 7" in x for x in why))

    def test_missing_config_has_reasons_instead_of_assertion(self):
        self.assertTrue(config_disable_reasons(NS(text_config=NS()), 4, 4))


class _Rows:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class TestGenericView(TestCase):
    def test_qsa_metadata_comes_from_sparse_backend_when_view_metadata_is_none(self):
        meta = NS(is_cuda_graph=True, sequence_lengths=_Rows(4),
                  row_req_pool_indices=_Rows(4), indexer_metadata=NS(
                      decode_logical_positions=_Rows(4), defer_block_expansion=True))
        sparse = NS(forward_metadata=meta, compress_ratio=4)
        st = NS(view=NS(metadata=None), fb=NS(
            forward_mode=NS(is_target_verify=lambda: True), req_pool_indices=_Rows(1),
            out_cache_loc=_Rows(4), spec_info=NS(topk=1)))
        self.assertEqual(metadata_reasons(NS(), st, sparse), [])
        sparse.forward_metadata = None
        self.assertIn("QSA graph metadata", metadata_reasons(NS(), st, sparse)[0])

    def test_prefill_is_rejected_before_reading_other_metadata(self):
        st = NS(fb=NS(forward_mode=NS(is_target_verify=lambda: False)))
        self.assertTrue(metadata_reasons(NS(), st, NS()))
