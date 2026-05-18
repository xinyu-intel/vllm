# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU Attention backend with split prefill/decode dispatch.

Inspired by ROCm AITER FA and FlashInfer backends, this backend splits
batches into decode and prefill portions and dispatches specialized kernel
configurations for each. Host-side metadata (query lengths, kv cache lengths)
is computed on CPU to enable better kernel dispatch decisions without
GPU-CPU synchronization overhead.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class XPUAttentionDecodeMetadata:
    max_query_len: int
    max_seq_len: int
    num_decodes: int


@dataclass
class XPUAttentionPrefillMetadata:
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    num_prefills: int


@dataclass
class XPUAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool

    # Split metadata
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode_metadata: XPUAttentionDecodeMetadata | None
    prefill_metadata: XPUAttentionPrefillMetadata | None

    # For cascade attention
    use_cascade: bool
    common_prefix_len: int


class XPUAttentionMetadataBuilder(AttentionMetadataBuilder[XPUAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XPUAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        assert self.reorder_batch_threshold is not None
        (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
        ) = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        decode_metadata = None
        if num_decodes > 0:
            decode_max_query_len = int(query_lens_cpu[:num_decodes].max().item())
            decode_max_seq_len = max_seq_len
            decode_metadata = XPUAttentionDecodeMetadata(
                max_query_len=decode_max_query_len,
                max_seq_len=decode_max_seq_len,
                num_decodes=num_decodes,
            )

        prefill_metadata = None
        if num_prefills > 0:
            prefill_query_lens = query_lens_cpu[num_decodes:]
            prefill_max_query_len = int(prefill_query_lens.max().item())
            prefill_query_start_loc = query_start_loc[num_decodes:]
            prefill_query_start_loc = (
                prefill_query_start_loc - prefill_query_start_loc[0]
            )
            if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
                prefill_max_seq_len = int(
                    common_attn_metadata.seq_lens_cpu_upper_bound[num_decodes:]
                    .max()
                    .item()
                )
            else:
                prefill_max_seq_len = max_seq_len
            prefill_metadata = XPUAttentionPrefillMetadata(
                max_query_len=prefill_max_query_len,
                max_seq_len=prefill_max_seq_len,
                query_start_loc=prefill_query_start_loc,
                num_prefills=num_prefills,
            )

        use_cascade = common_prefix_len > 0

        return XPUAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=causal,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class XPUAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return max(default_block_size, 64)

    @staticmethod
    def get_name() -> str:
        return "XPU_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_impl_cls() -> type["XPUAttentionImpl"]:
        return XPUAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["XPUAttentionMetadataBuilder"]:
        return XPUAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return True
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return False


class XPUAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type

        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            reshape_and_cache_flash,
        )

        self._flash_attn_varlen_func = flash_attn_varlen_func
        self._reshape_and_cache_flash = reshape_and_cache_flash

    def _decode_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: XPUAttentionMetadata,
    ) -> None:
        """Decode path: single-token generation, memory-bandwidth bound.

        For decode, each request generates one token and must load the
        entire KV cache. We use flash_attn_varlen_func with max_seqlen_q=1
        which triggers optimized decode-specific code paths in the kernel.
        """
        assert attn_metadata.decode_metadata is not None
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_query = query[:num_decode_tokens]
        decode_seq_lens = attn_metadata.seq_lens[:num_decodes]
        decode_block_table = attn_metadata.block_table[:num_decodes]
        decode_cu_seqlens_q = attn_metadata.query_start_loc[: num_decodes + 1]

        self._flash_attn_varlen_func(
            q=decode_query,
            k=key_cache,
            v=value_cache,
            out=output[:num_decode_tokens],
            cu_seqlens_q=decode_cu_seqlens_q,
            max_seqlen_q=attn_metadata.decode_metadata.max_query_len,
            seqused_k=decode_seq_lens,
            max_seqlen_k=attn_metadata.decode_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=list(self.sliding_window),
            block_table=decode_block_table,
            softcap=self.logits_soft_cap,
            fa_version=2,
        )

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: XPUAttentionMetadata,
    ) -> None:
        """Prefill path: process new prompt tokens, compute-bound.

        For prefill/extend, we have multiple query tokens per request
        that need to attend to both the existing KV cache and the new
        K/V from the current batch. We use the paged KV cache path.
        """
        assert attn_metadata.prefill_metadata is not None
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills

        prefill_query = query[num_decode_tokens:]
        prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
        prefill_block_table = attn_metadata.block_table[
            num_decodes : num_decodes + num_prefills
        ]

        self._flash_attn_varlen_func(
            q=prefill_query,
            k=key_cache,
            v=value_cache,
            out=output[num_decode_tokens:],
            cu_seqlens_q=attn_metadata.prefill_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.prefill_metadata.max_query_len,
            seqused_k=prefill_seq_lens,
            max_seqlen_k=attn_metadata.prefill_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=list(self.sliding_window),
            block_table=prefill_block_table,
            softcap=self.logits_soft_cap,
            fa_version=2,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XPUAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with split decode/prefill dispatch.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for XPUAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention (no KV cache)
        if self.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
            )

        key_cache, value_cache = kv_cache.unbind(0)

        query = query[:num_actual_tokens]
        output_slice = output[:num_actual_tokens]

        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills

        if num_decodes > 0 and num_prefills > 0:
            # Mixed batch: dispatch decode and prefill separately
            self._decode_attention(
                query, key_cache, value_cache, output_slice, attn_metadata
            )
            self._prefill_attention(
                query,
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                key_cache,
                value_cache,
                output_slice,
                attn_metadata,
            )
        elif num_decodes > 0:
            # Pure decode batch
            self._decode_attention(
                query, key_cache, value_cache, output_slice, attn_metadata
            )
        elif num_prefills > 0:
            # Pure prefill batch
            self._prefill_attention(
                query,
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                key_cache,
                value_cache,
                output_slice,
                attn_metadata,
            )
        else:
            # Unified fallback (e.g., during profiling)
            self._flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                out=output_slice,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=list(self.sliding_window),
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                fa_version=2,
            )

        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ):
            return

        key_cache, value_cache = kv_cache.unbind(0)
        self._reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: XPUAttentionMetadata,
    ) -> torch.Tensor:
        """Encoder attention without KV cache (bidirectional)."""
        self._flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=list(self.sliding_window),
            softcap=self.logits_soft_cap,
            fa_version=2,
        )
        return output
