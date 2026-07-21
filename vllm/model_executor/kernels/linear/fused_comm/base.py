# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch


class FusedCommLinearKernel(ABC):
    """Fused collective + GEMM kernel for sequence parallelism.

    Each instance is bound to a specific linear layer after weight loading.
    The kernel does not own or modify weights — it relies on the underlying
    quant method's process_weights_after_loading having already set the
    correct weight layout.
    """

    @classmethod
    @abstractmethod
    def is_supported(cls) -> tuple[bool, str | None]:
        raise NotImplementedError

    @abstractmethod
    def fused_ag_gemm(
        self,
        layer: torch.nn.Module,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        """AllGather + GEMM: [N/tp, H_in] -> [N, H_out_partition]"""
        raise NotImplementedError

    @abstractmethod
    def fused_gemm_rs(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        """GEMM + ReduceScatter: [N, H_in_partition] -> [N/tp, H_out]"""
        raise NotImplementedError
