# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.kernels.linear import (  # noqa: E501
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    MXFP8ScaledMMLinearKernel,
    MXFP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass


class XPUMXFP8ScaledMMLinearKernel(MXFP8ScaledMMLinearKernel):
    """
    MXFP8 linear kernels using Torch.
    """

    @classmethod
    def can_implement(
        cls, c: MXFP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        in_features, out_features = c.partition_weight_shape
        if in_features % MXFP8_BLOCK_SIZE or out_features % MXFP8_BLOCK_SIZE:
            return (
                False,
                f"XPU MXFP8 Linear requires in/out features to be multiples of "
                f"{MXFP8_BLOCK_SIZE}, got in_features={in_features}, "
                f"out_features={out_features}",
            )

        return True, None

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "requires XPU."

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_scale = layer.weight_scale.view(torch.float8_e8m0fnu)
        weight_scale = weight_scale.t().contiguous()
        replace_parameter(layer, "weight", layer.weight.t())
        replace_parameter(layer, "weight_scale", weight_scale.data)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        output = torch.ops._xpu_C.fp8_gemm(A, B, out_dtype, As, Bs, bias)
        return torch.narrow(output, 0, 0, output_shape[0]).view(*output_shape)
