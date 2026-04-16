# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
    kMxfp4Static,
)
from vllm.platforms import current_platform

if current_platform.is_xpu():
    from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe


class XPUExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.is_fp8 = False
        self.is_mxfp4 = False

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        topk = topk_ids.size(-1)
        xpu_fused_moe(
            hidden_states=hidden_states,
            w13=w1,
            w13_scales=self.w1_scale,
            w13_bias=self.w1_bias,
            w2=w2,
            w2_scales=self.w2_scale,
            w2_bias=self.w2_bias,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            n_experts_per_token=topk,
            activation=activation.value,
            num_experts=self.moe_config.num_local_experts,
            ep_rank=self.moe_config.ep_rank,
            ep_size=self.moe_config.ep_size,
            output=output,
            is_fp8=self.is_fp8,
            is_mxfp4=self.is_mxfp4,
        )


class XPUExpertsFp8(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.is_fp8 = True


class XPUExpertsMXFp4(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.is_mxfp4 = True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


def xpu_batched_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    activation: MoEActivation,
    output: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
) -> None:
    """
    Batched MoE forward using XPU grouped GEMM.

    Args:
        hidden_states: [E, padded_M, hidden_size] - batched input per expert
        w1: gate+up weights (same format as standard XPU path)
        w2: down weights (same format as standard XPU path)
        expert_num_tokens: [E] int32 - actual token count per expert
        activation: activation function
        output: [E, padded_M, hidden_size] - output tensor
        workspace13: workspace for gemm1 output and gemm2 input
        workspace2: workspace for activation output
    """
    E = hidden_states.size(0)
    padded_M = hidden_states.size(1)
    hidden_size = hidden_states.size(2)
    device = hidden_states.device

    # Use same dimension derivation as xpu_fused_moe standard path:
    # w1.shape[-1] // 2 gives inter_size for bf16 weights
    inter_size = w1.shape[-1] // 2

    # Compute padded expert_first_token_offset: [0, pM, 2*pM, ..., E*pM]
    expert_first_token_offset = (
        torch.arange(0, E + 1, device=device, dtype=torch.int64) * padded_M
    )

    # Flatten to 2D for grouped GEMM
    flat_input = hidden_states.reshape(E * padded_M, hidden_size)

    # GEMM1: input x w1 -> [E*padded_M, 2*inter_size]
    mm1_out = _resize_cache(workspace13, (E * padded_M, 2 * inter_size))

    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=flat_input,
        ptr_B=w1,
        ptr_scales=None,
        ptr_bias=None,
        ptr_D=mm1_out,
        expert_first_token_offset=expert_first_token_offset,
        N=2 * inter_size,
        K=hidden_size,
        num_experts=E,
        is_B_int4=False,
        is_B_mxfp4=False,
        expert_num_tokens=expert_num_tokens,
    )

    # Activation
    act_out = _resize_cache(workspace2, (E * padded_M, inter_size))
    apply_moe_activation(activation, act_out, mm1_out)

    # GEMM2: act_out x w2 -> [E*padded_M, hidden_size]
    mm2_out = _resize_cache(workspace13, (E * padded_M, hidden_size))

    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=act_out,
        ptr_B=w2,
        ptr_scales=None,
        ptr_bias=None,
        ptr_D=mm2_out,
        expert_first_token_offset=expert_first_token_offset,
        N=hidden_size,
        K=inter_size,
        num_experts=E,
        is_B_int4=False,
        is_B_mxfp4=False,
        expert_num_tokens=expert_num_tokens,
    )

    output.copy_(mm2_out.reshape(E, padded_M, hidden_size), non_blocking=True)


class XPUBatchedExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
        ]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (None, None)

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        num_dp = self.num_dispatchers
        assert num_dp is not None
        experts_per_worker = self.moe_config.num_local_experts
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (experts_per_worker, M * num_dp, max(N, K))
        workspace2 = (
            experts_per_worker,
            M * num_dp,
            max(activation_out_dim, K),
        )
        output = (experts_per_worker, M * num_dp, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        xpu_batched_fused_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            expert_num_tokens=expert_num_tokens,
            activation=activation,
            output=output,
            workspace13=workspace13,
            workspace2=workspace2,
        )
