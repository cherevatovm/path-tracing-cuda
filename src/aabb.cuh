#pragma once

#include "vec3.cuh"
#include "ray.cuh"
#include "scene.cuh"

struct AABB {
    Vec3 min, max;

    __host__ __device__ AABB() {}
    __host__ __device__ AABB(const Vec3& mn, const Vec3& mx) : min(mn), max(mx) {}

    __host__ __device__ bool isEmpty() const {
        return min.x >= max.x || min.y >= max.y || min.z >= max.z;
    }

    __host__ __device__ Vec3 centroid() const { return (min + max) * 0.5f; }

    __host__ __device__ int largestAxis() const {
        Vec3 d = max - min;
        if (d.x > d.y && d.x > d.z) return 0;
        return (d.y > d.z) ? 1 : 2;
    }

    __host__ __device__ bool intersect(
        const Ray& r,
        const Vec3& invDir,
        float tMax,
        float& tMinOut
    ) const {
        float t0 = 0.f, t1 = tMax;

        for (int i = 0; i < 3; ++i) {
            float tNear = (min[i] - r.orig[i]) * invDir[i];
            float tFar  = (max[i] - r.orig[i]) * invDir[i];

            if (tNear > tFar) {
                float tmp = tNear; tNear = tFar; tFar = tmp;
            }

            t0 = fmaxf(t0, tNear);
            t1 = fminf(t1, tFar);
            if (t0 > t1) return false;
        }

        tMinOut = t0;
        return true;
    }

    __host__ __device__ bool intersect(const Ray& r, const Vec3& invDir, float tMax) const {
        float t0 = 0.f, t1 = tMax;

        for (int i = 0; i < 3; ++i) {
            float tNear = (min[i] - r.orig[i]) * invDir[i];
            float tFar  = (max[i] - r.orig[i]) * invDir[i];

            if (tNear > tFar) {
                float tmp = tNear; tNear = tFar; tFar = tmp;
            }

            t0 = fmaxf(t0, tNear);
            t1 = fminf(t1, tFar);
            if (t0 > t1) return false;
        }

        return true;
    }

    __host__ __device__ static AABB empty() {
        return AABB(
            Vec3(1e30f, 1e30f, 1e30f),
            Vec3(-1e30f, -1e30f, -1e30f)
        );
    }

    __host__ __device__ static AABB unite(const AABB& a, const AABB& b) {
        return AABB(
            Vec3(fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z)),
            Vec3(fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z))
        );
    }

    __host__ __device__ static AABB unite(const AABB& a, const Vec3& p) {
        return AABB(
            Vec3(fminf(a.min.x, p.x), fminf(a.min.y, p.y), fminf(a.min.z, p.z)),
            Vec3(fmaxf(a.max.x, p.x), fmaxf(a.max.y, p.y), fmaxf(a.max.z, p.z))
        );
    }
};

inline AABB shapeAABB(const Shape& s) {
    const float eps = 1e-4f;
    if (s.type == SHAPE_SPHERE) {
        float r = s.sphere.radius;
        Vec3 c = s.sphere.center;
        return AABB(
            Vec3(c.x - r - eps, c.y - r - eps, c.z - r - eps),
            Vec3(c.x + r + eps, c.y + r + eps, c.z + r + eps)
        );
    } else {
        Vec3 v0 = s.tri.v0, v1 = s.tri.v1, v2 = s.tri.v2;
        return AABB(
            Vec3(
                fminf(v0.x, fminf(v1.x, v2.x)) - eps,
                fminf(v0.y, fminf(v1.y, v2.y)) - eps,
                fminf(v0.z, fminf(v1.z, v2.z)) - eps
            ),
            Vec3(
                fmaxf(v0.x, fmaxf(v1.x, v2.x)) + eps,
                fmaxf(v0.y, fmaxf(v1.y, v2.y)) + eps,
                fmaxf(v0.z, fmaxf(v1.z, v2.z)) + eps
            )
        );
    }
}
