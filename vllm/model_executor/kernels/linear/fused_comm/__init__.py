# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from .base import FusedCommLinearKernel

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )

logger = init_logger(__name__)


def init_fused_comm_kernel(
    quant_method: QuantizeMethodBase,
) -> FusedCommLinearKernel | None:
    """Select and instantiate a FusedCommLinearKernel for the given layer.

    Returns None if fused comm is not enabled in config, or no fused
    implementation is available for the quant method / platform combination.
    """
    from vllm.config import get_current_vllm_config

    parallel_config = get_current_vllm_config().parallel_config
    if not (
        parallel_config.enable_sequence_parallel_fuse_gemm_comms
        and parallel_config.use_sequence_parallel
    ):
        return None

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    if isinstance(quant_method, UnquantizedLinearMethod):
        from .symm_mem import SymmMemFusedCommKernel

        supported, reason = SymmMemFusedCommKernel.is_supported()
        if supported:
            return SymmMemFusedCommKernel()
        logger.debug("Fused comm BF16 not available: %s", reason)
        return None

    if isinstance(quant_method, Fp8LinearMethod):
        fp8_kernel = quant_method.fp8_linear
        from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
            FP8ScaledMMLinearKernel,
        )

        if not isinstance(fp8_kernel, FP8ScaledMMLinearKernel):
            logger.debug(
                "Fused comm FP8 not available: kernel %s is not "
                "FP8ScaledMMLinearKernel",
                type(fp8_kernel).__name__,
            )
            return None

        from .symm_mem import SymmMemScaledFusedCommKernel

        supported, reason = SymmMemScaledFusedCommKernel.is_supported()
        if supported:
            return SymmMemScaledFusedCommKernel(fp8_kernel)
        logger.debug("Fused comm FP8 not available: %s", reason)
        return None

    logger.debug(
        "Fused comm not available for quant method %s",
        type(quant_method).__name__,
    )
    return None


__all__ = [
    "FusedCommLinearKernel",
    "init_fused_comm_kernel",
]
