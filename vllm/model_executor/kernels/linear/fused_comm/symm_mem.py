# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
)

from .base import FusedCommLinearKernel


class SymmMemFusedCommKernel(FusedCommLinearKernel):
    """Unquantized fused comm+GEMM using torch symmetric memory."""

    @classmethod
    def is_supported(cls) -> tuple[bool, str | None]:
        from vllm.platforms import current_platform

        if not current_platform.is_cuda_alike():
            return False, "symmetric memory requires CUDA"
        return True, None

    def fused_ag_gemm(
        self,
        layer: torch.nn.Module,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        from torch.distributed._symmetric_memory import (
            enable_symm_mem_for_group,
        )

        enable_symm_mem_for_group(group_name)
        W = layer.weight.t()
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard, [W], gather_dim=0, group_name=group_name
        )
        output = mm_outputs[0]
        if bias is not None:
            output = output + bias
        return output

    def fused_gemm_rs(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        from torch.distributed._symmetric_memory import (
            enable_symm_mem_for_group,
        )

        enable_symm_mem_for_group(group_name)
        W = layer.weight.t()
        output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x, W, "sum", scatter_dim=0, group_name=group_name
        )
        if bias is not None:
            output = output + bias
        return output


class SymmMemScaledFusedCommKernel(FusedCommLinearKernel):
    """FP8 W8A8 fused comm+GEMM using torch symmetric memory.

    Requires the underlying FP8 kernel to provide quant_fp8 for
    activation quantization. Weight must already be in [K, N] layout
    (handled by the kernel's process_weights_after_loading).
    """

    def __init__(self, fp8_kernel: FP8ScaledMMLinearKernel) -> None:
        self._fp8_kernel = fp8_kernel

    @classmethod
    def is_supported(cls) -> tuple[bool, str | None]:
        from vllm.platforms import current_platform

        if not current_platform.is_cuda_alike():
            return False, "symmetric memory requires CUDA"
        return True, None

    def fused_ag_gemm(
        self,
        layer: torch.nn.Module,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        from torch.distributed._symmetric_memory import (
            enable_symm_mem_for_group,
        )

        enable_symm_mem_for_group(group_name)

        x_2d = x_shard.view(-1, x_shard.shape[-1])
        w, w_scale, x_scale, x_scale_ub = self._fp8_kernel._get_layer_params(layer)
        x_q, a_scale = self._fp8_kernel.quant_fp8(x_2d, x_scale, x_scale_ub)

        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            x_q,
            [w],
            a_scale,
            [w_scale],
            gather_dim=0,
            biases=[bias],
            result_scales=[None],
            out_dtypes=[torch.bfloat16],
            use_fast_accum=[False],
            group_name=group_name,
        )
        return mm_outputs[0]

    def fused_gemm_rs(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        from torch.distributed._symmetric_memory import (
            enable_symm_mem_for_group,
        )

        enable_symm_mem_for_group(group_name)

        x_2d = x.view(-1, x.shape[-1])
        w, w_scale, x_scale, x_scale_ub = self._fp8_kernel._get_layer_params(layer)
        x_q, a_scale = self._fp8_kernel.quant_fp8(x_2d, x_scale, x_scale_ub)

        output_shape = [x_2d.shape[0], w.shape[1]]
        return torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
            x_q,
            w,
            a_scale,
            w_scale,
            "sum",
            0,
            0,
            group_name,
            output_shape,
            bias,
            None,
            torch.bfloat16,
            False,
        )
