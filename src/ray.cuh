#pragma once

#include "vec3.cuh"

struct Ray {
    Vec3 orig;
    Vec3 dir;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3& o, const Vec3& d) : orig(o), dir(d) {}
};