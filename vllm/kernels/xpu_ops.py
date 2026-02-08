# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

import torch
from torch import Tensor
from torch.library import Library

from vllm import ir
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

current_platform.import_kernels()


def is_xpu_kernels_found() -> bool:
    from importlib.util import find_spec

    return find_spec("vllm_xpu_kernels") is not None


xpu_kernels_lib = Library("xpu_kernels", "FRAGMENT")
"""
This library holds torch custom ops for wrapped vLLM XPU kernels.
Many vLLM XPU kernels want to remain invisible to torch.compile even after lowering.
They are thus wrapped into torch custom ops inside the IR op implementations.
"""

direct_register_xpu_kernels_op = functools.partial(
    direct_register_custom_op, target_lib=xpu_kernels_lib
)
"""Syntactic sugar for registering vLLM XPU kernels custom ops."""

XPU_KERNELS_SUPPORTED = is_xpu_kernels_found()
"""Most kernels in this file are supported if vLLM XPU kernels are installed."""

rms_no_var = lambda x, w, e, v=None: v is None


@ir.ops.rms_norm.register_impl(
    "xpu_kernels", supports_args=rms_no_var, supported=XPU_KERNELS_SUPPORTED
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    if weight is None:
        # Kernel requires weight tensor, pass ones
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    assert variance_size is None
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output
