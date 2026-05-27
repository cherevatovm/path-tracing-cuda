#pragma once

#include "bvh.cuh"

extern "C" __global__
void renderKernel(
    Vec3* framebuffer,
    const BVHNode* bvhNodes,
    const Shape* shapes,
    int shapeCount,
    int width,
    int height,
    int spp
);