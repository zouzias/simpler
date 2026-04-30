#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test triangular inverse cube-only method: runtime-configurable  C = torch.linalg.triangular_solve(A, torch.eye(A.shape[-1]))."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestTriangularInverse(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/triangular_inverse_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "GEMM",
                "source": "kernels/aic/kernel_simple_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            }
        ],
    }

    CASES = [
        {
            "name": "Case0",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 20},
            "params": {"matmul_add_task_num": 1, "incore_data_size": 128, "incore_loop": 1, "grid_k": 1},
        },
        {
            "name": "Case1",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 20},
            "params": {"matmul_add_task_num": 1, "incore_data_size": 128, "incore_loop": 1, "grid_k": 1},
        },
        {
            "name": "Case2",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 20},
            "params": {"matmul_add_task_num": 1, "incore_data_size": 128, "incore_loop": 1, "grid_k": 1},
        },
    ]

    def generate_args(self, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = torch.randn(tile_size, tile_size, dtype=torch.float32) * 0.1
        B = torch.randn(tile_size, tile_size, dtype=torch.float32) * 0.1
        C = torch.zeros(tile_size, tile_size, dtype=torch.float32)
        config = torch.tensor([tile_size, grid_k, num_groups, incore_loop], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("A", A.flatten()), Tensor("B", B.flatten()), Tensor("C", C.flatten()), Tensor("config", config)
        )

    def compute_golden(self, args, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = args.A.reshape(tile_size, tile_size)
        B = args.B.reshape(tile_size, tile_size)
        C = args.C.reshape(tile_size, tile_size)
        C[:] = 0.0
        C = A @ B


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
