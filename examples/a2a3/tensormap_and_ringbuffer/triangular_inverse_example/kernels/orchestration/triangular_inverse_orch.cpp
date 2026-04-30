/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * BGEMM Orchestration Function (tensormap_and_ringbuffer Runtime)
 *
 * Builds the task graph for tiled matrix multiplication: C = A @ B
 *
 * Configuration read from scalar args (set in golden.py):
 *   - tile_size: tile dimension (tile_size x tile_size per tile)
 *   - grid_k: number of K-dimension partitions
 *   - num_groups: number of independent groups (= matmul_add_task_num / grid_k)
 *   - incore_loop: number of tiles per group
 *
 * Memory layout (tile-first, flattened):
 *   A: [num_groups, grid_k, incore_loop, tile_size, tile_size]
 *   B: [num_groups, grid_k, incore_loop, tile_size, tile_size]
 *   C: [incore_loop * num_groups, tile_size, tile_size]
 *
 * Arg layout: [A, B, C, config]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    // Tensor args
    Tensor ext_A = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_B = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_C = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_config = from_tensor_arg(orch_args.tensor(3));

    // Read config from tensor data: [tile_size, grid_k, num_groups, incore_loop]
    int64_t *host_config = orch_args.tensor(3).data_as<int64_t>();
    int tile_size = static_cast<int>(host_config[0]);
    int grid_k = static_cast<int>(host_config[1]);
    int num_groups = static_cast<int>(host_config[2]);
    int incore_loop = static_cast<int>(host_config[3]);
    uint64_t tile_elems = static_cast<uint64_t>(tile_size) * tile_size;

    int grid_m = 1;
    int grid_n = 1;

    LOG_INFO_V0(
        "[bgemm_orch] tile_size: %d, grid_m: %d, grid_n: %d, grid_k: %d, num_groups: %d, incore_loop: %d", tile_size,
        grid_m, grid_n, grid_k, num_groups, incore_loop
    );

    uint32_t tile_shapes[1] = {static_cast<uint32_t>(tile_elems)};
    uint64_t group_tile_elems = static_cast<uint64_t>(incore_loop) * tile_elems;
    uint32_t group_shapes[1] = {static_cast<uint32_t>(group_tile_elems)};
    TensorCreateInfo group_ci(group_shapes, 1, DataType::FLOAT32);

    Arg params_gemm;
    params_gemm.add_input(ext_A);
    params_gemm.add_input(ext_B);
    params_gemm.add_output(ext_C);
    params_gemm.add_input(ext_config);
    TaskOutputTensors gemm_outs = rt_submit_aic_task(FUNC_GEMM_TILE, params_gemm);
}

}  // extern "C"
