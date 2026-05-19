#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Triangular inverse (recursive unrolled): M_inv = inv(M) for upper/lower-triangular fp16 matrices."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestTriangularInverse(SceneTestCase):
    # fp16 arithmetic — use tolerances appropriate for half-precision results
    RTOL = 1e-2
    ATOL = 1e-2

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/triangular_inverse_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "TRI_INV",
                "source": "kernels/aic/kernel_tri_inv_rec_unroll.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT, D.IN],
            }
        ],
    }

    CASES = [
        {
            "name": "Case0_upper32",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 16},
            "params": {"num_matrices": 4, "matrix_size": 32, "is_lower": 0},
        },
        {
            "name": "Case1_upper64",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 16},
            "params": {"num_matrices": 4, "matrix_size": 64, "is_lower": 0},
        },
        {
            "name": "Case2_lower32",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 16},
            "params": {"num_matrices": 4, "matrix_size": 32, "is_lower": 1},
        },
    ]

    def generate_args(self, params):
        n = params["matrix_size"]
        num_matrices = params["num_matrices"]
        is_lower = params["is_lower"]

        # Build well-conditioned triangular matrices in fp16.
        # Start with random values and zero out the off-triangle, then set
        # the diagonal to a value in [0.5, 1.5] to ensure invertibility.
        M_f32 = torch.rand(num_matrices, n, n, dtype=torch.float32)
        if is_lower:
            M_f32 = torch.tril(M_f32)
        else:
            M_f32 = torch.triu(M_f32)
        # Diagonal entries in [0.5, 1.5] — keeps condition number reasonable
        diag_vals = torch.rand(num_matrices, n, dtype=torch.float32) + 0.5
        idx = torch.arange(n)
        M_f32[:, idx, idx] = diag_vals
        M = M_f32.to(torch.float16)

        # Negative identity matrix — used by the kernel to derive I and Zero
        I_neg = (-torch.eye(n, dtype=torch.float16)).flatten()

        M_inv = torch.zeros(num_matrices, n, n, dtype=torch.float16)
        config = torch.tensor([n, num_matrices, is_lower], dtype=torch.int64)

        return TaskArgsBuilder(
            Tensor("M", M.flatten()),
            Tensor("I_neg", I_neg),
            Tensor("M_inv", M_inv.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        n = params["matrix_size"]
        num_matrices = params["num_matrices"]
        M = args.M.reshape(num_matrices, n, n).to(torch.float32)
        # Compute in float32 for numerical stability, then cast to fp16
        M_inv = torch.linalg.inv(M).to(torch.float16)
        return M_inv


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
