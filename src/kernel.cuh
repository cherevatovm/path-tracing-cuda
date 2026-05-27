#pragma once

#include "scene.cuh"

extern "C" __global__
void renderKernel(
    Vec3* framebuffer,
    const Shape* shapes,
    int shapeCount,
    int width,
    int height,
    int spp
);