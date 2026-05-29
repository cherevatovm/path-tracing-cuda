#pragma once

#include "bvh.cuh"

extern "C" __global__
__launch_bounds__(256, 2)
void renderKernel(
    Vec3* framebuffer,
    const BVHNode* bvhNodes,
    const Shape* shapes,
    int shapeCount,
    const int* lightIndices,
    int lightCount,
    int width,
    int height,
    int spp
);