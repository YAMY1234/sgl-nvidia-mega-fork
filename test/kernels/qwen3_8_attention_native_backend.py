"""Verbatim main4517 native sparse decode path, standalone numerical harness.

Only the gather/gate launch wrappers are supplied with their CPU-compiled
native cubins; TRTLLM and the backend method body are unchanged.
"""
from typing import Tuple
import torch

# BF16 inputs never take this FP8-only conditional.
def is_sm100_supported():
    return torch.cuda.get_device_capability()[0] == 10

_TRTLLM_SPARSE_PAGE_SIZE = 64
fused_kv_prepare = None

class NativeQSA:
    def _forward_trtllm_sparse(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        layer,
        forward_batch,
        metadata,
        topk_indices: torch.Tensor,
        trtllm_decode,
        new_kv=None,
    ) -> torch.Tensor:
        """Pack selected KV at page-aligned row strides for FlashInfer's paged decode,
        driven by a static arange block table and the per-row valid counts."""
        batch, topk = topk_indices.shape
        compress_ratio = (
            self.compress_ratio if self._uses_block_indices(topk_indices) else 1
        )
        topk = topk * compress_ratio + compress_ratio - 1
        page = _TRTLLM_SPARSE_PAGE_SIZE
        pages_per_row = (topk + page - 1) // page
        stride = pages_per_row * page
        device = q.device
        sequence_lens = metadata.sequence_lengths
        if metadata.is_cuda_graph:
            valid_counts = metadata.fa2_valid_counts
            if valid_counts is None:
                raise RuntimeError("QSA CUDA graph metadata is incomplete")
        else:
            valid_counts = torch.empty(batch, dtype=torch.int32, device=device)
        cu_strided, block_tables = self._get_trtllm_sparse_tables(
            batch, pages_per_row, page, device
        )
        capacity_rows = self._cuda_graph_max_tokens if metadata.is_cuda_graph else batch
        # TRTLLM-gen takes BF16 q with FP8-E4M3 KV, so an FP8 pool stays FP8 here;
        # validated on SM10x only, SM120 keeps the BF16 scratch.
        packed_dtype = (
            k_buffer.dtype
            if k_buffer.dtype == torch.float8_e4m3fn and is_sm100_supported()
            else q.dtype
        )
        packed_k, packed_v = self._get_fa2_scratch(
            max(capacity_rows, batch) * stride,
            k_buffer.shape[1],
            k_buffer.shape[2],
            packed_dtype,
            k_buffer.device,
        )
        if new_kv is not None:
            k, v, loc, width = new_kv
            row_requests = metadata.row_req_pool_indices
            if row_requests is None:
                row_requests = forward_batch.req_pool_indices
            fused_kv_prepare(
                k_buffer,
                v_buffer,
                k,
                v,
                loc,
                self.req_to_token_pool.req_to_token,
                row_requests,
                topk_indices,
                sequence_lens,
                valid_counts,
                packed_k[: batch * stride],
                packed_v[: batch * stride],
                width,
                compress_ratio=compress_ratio,
                chain_positions=True,
                query_positions=(
                    metadata.indexer_metadata.decode_logical_positions
                    if compress_ratio > 1
                    else None
                ),
                # trtllm_decode reads each row only up to valid_counts.
                visible_page_size=page,
            )
        else:
            qwen_sparse_valid_counts_triton(
                sequence_lens, topk_indices, valid_counts, batch, topk
            )
            qwen_sparse_kv_extraction_compact_triton(
                k_buffer,
                v_buffer,
                self.req_to_token_pool.req_to_token,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_strided,
                packed_k,
                packed_v,
                batch,
                topk,
                zero_fill_cols=stride,
            )
        num_kv_heads = k_buffer.shape[1]
        head_dim = k_buffer.shape[2]
        kc = (
            packed_k[: batch * stride]
            .view(-1, page, num_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        vc = (
            packed_v[: batch * stride]
            .view(-1, page, num_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        if self._trtllm_workspace is None:
            self._trtllm_workspace = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

        key = (batch, q.shape[1], device)
        if key not in self._trtllm_counters:
            size = get_trtllm_gen_multi_ctas_kv_counter_bytes(
                batch,
                q.shape[1],
                torch.cuda.get_device_properties(device).multi_processor_count,
            )
            self._trtllm_counters[key] = torch.zeros(
                size, dtype=torch.uint8, device=device
            )
        output = trtllm_decode(
            query=q.contiguous(),
            kv_cache=(kc, vc),
            workspace_buffer=self._trtllm_workspace,
            block_tables=block_tables,
            seq_lens=valid_counts,
            max_seq_len=stride,
            bmm1_scale=layer.scaling,
            bmm2_scale=1.0,
            multi_ctas_kv_counter_buffer=self._trtllm_counters[key],
        )
        return output.reshape(q.shape[0], -1)


    def _get_fa2_scratch(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (num_kv_heads, head_dim, dtype, device)
        buffers = self._fa2_scratch.get(key)
        if buffers is None or buffers[0].shape[0] < capacity:
            shape = (capacity, num_kv_heads, head_dim)
            # Zeroed once: a row is packed only up to its last visible page, so
            # bytes past it are never written and must not be NaN garbage.
            buffers = (
                torch.zeros(shape, dtype=dtype, device=device),
                torch.zeros(shape, dtype=dtype, device=device),
            )
            self._fa2_scratch[key] = buffers
        return buffers[0][:capacity], buffers[1][:capacity]


    def _get_trtllm_sparse_tables(self, batch, pages_per_row, page, device):
        key = (batch, pages_per_row, device)
        cached = self._trtllm_sparse_tables.get(key)
        if cached is None:
            stride = pages_per_row * page
            cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * stride
            block_tables = (
                torch.arange(batch, dtype=torch.int32, device=device)[:, None]
                * pages_per_row
                + torch.arange(pages_per_row, dtype=torch.int32, device=device)[None, :]
            ).contiguous()
            cached = (cu, block_tables)
            self._trtllm_sparse_tables[key] = cached
        return cached


    def _uses_block_indices(self, indices):
        return (
            self.qsa_profile is not None
            and indices.shape[1] == self.qsa_profile.block_topk
        )
