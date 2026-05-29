#pragma once

#include "vec3.cuh"
#include "ray.cuh"

constexpr float EPS = 1e-4f;

enum ReflType { DIFF, SPEC, REFR };
enum ShapeType { SHAPE_SPHERE, SHAPE_TRIANGLE };

struct Sphere {
    Vec3 center;
    float radius;
};

struct Triangle {
    Vec3 v0, v1, v2;
    Vec3 normal;
    Vec3 vn0, vn1, vn2;
};

struct Shape {
    ShapeType type;
    Vec3 emis;
    Vec3 color;
    ReflType refl;

    Sphere sphere;
    Triangle tri;
};

struct Hit {
    float t;
    int id;
    Vec3 n;
};

__host__ __device__ inline void create_orthonorm_sys(const Vec3& v1, Vec3& v2, Vec3& v3) {
    if (fabsf(v1.x) > fabsf(v1.y)) {
        float inv_len = 1.0f / sqrtf(v1.x * v1.x + v1.z * v1.z);
        v2 = Vec3(-v1.z * inv_len, 0.0f, v1.x * inv_len);
    } else {
        float inv_len = 1.0f / sqrtf(v1.y * v1.y + v1.z * v1.z);
        v2 = Vec3(0.0f, v1.z * inv_len, -v1.y * inv_len);
    }
    v3 = v1.cross(v2);
}