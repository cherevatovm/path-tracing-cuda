#pragma once

#include "aabb.cuh"
#include <vector>
#include <algorithm>

struct BVHNode {
    AABB bounds;
    int objCount;           // 0 = interior node, >0 = leaf with n primitives
    int splitAxis;          // 0/1/2 -> x/y/z (meaningful only for interior nodes)
    int firstObjOrChild;    // leaf: first primitive index; interior: right child index
};

struct BoundingInfo {
    AABB aabb;
    Vec3 centroid;
    int originalIdx;

    BoundingInfo() {}
    BoundingInfo(const AABB& b, int idx)
        : aabb(b), centroid(b.centroid()), originalIdx(idx) {}
};

struct CentroidComparator {
    int axis;
    CentroidComparator(int ax) : axis(ax) {}

    bool operator()(const BoundingInfo& a, const BoundingInfo& b) const {
        return a.centroid[axis] < b.centroid[axis];
    }
};

class BVHBuilder {
public:
    BVHBuilder() {}

    void buildInPlace(
        std::vector<Shape>& shapes,
        std::vector<BVHNode>& outNodes,
        int maxLeafSize = 4
    ) {
        const int n = (int)shapes.size();
        if (n == 0) {
            outNodes.clear();
            return;
        }

        std::vector<BoundingInfo> infos;
        infos.reserve(n);
        for (int i = 0; i < n; ++i) {
            AABB aabb = shapeAABB(shapes[i]);
            infos.emplace_back(aabb, i);
        }

        nodes_.clear();
        nodes_.reserve(n * 2);
        buildRecursive(infos, 0, n, maxLeafSize);

        std::vector<Shape> reordered;
        reordered.reserve(n);
        for (const auto& info : infos)
            reordered.push_back(shapes[info.originalIdx]);

        shapes.swap(reordered);
        outNodes.swap(nodes_);
    }

    static void build(
        std::vector<Shape>& shapes,
        std::vector<BVHNode>& outNodes,
        int maxLeafSize = 4
    ) {
        BVHBuilder b;
        b.buildInPlace(shapes, outNodes, maxLeafSize);
    }

private:
    std::vector<BVHNode> nodes_;

    void buildRecursive(
        std::vector<BoundingInfo>& infos,
        int begin, int end,
        int maxLeafSize
    ) {
        BVHNode node;
        int objCnt = end - begin;

        AABB rangeAABB = AABB::empty();
        for (int i = begin; i < end; ++i)
            rangeAABB = AABB::unite(rangeAABB, infos[i].aabb);

        if (objCnt <= maxLeafSize) {
            // leaf
            node.bounds = rangeAABB;
            node.objCount = objCnt;
            node.splitAxis = 0;
            node.firstObjOrChild = begin;
            nodes_.push_back(node);
            return;
        }

        // interior node
        AABB centroidAABB = AABB::empty();
        for (int i = begin; i < end; ++i)
            centroidAABB = AABB::unite(centroidAABB, infos[i].centroid);

        int axis = centroidAABB.largestAxis();
        int mid = begin + objCnt / 2;

        std::nth_element(
            infos.begin() + begin,
            infos.begin() + mid,
            infos.begin() + end,
            CentroidComparator(axis)
        );

        int nodeIdx = (int)nodes_.size();
        nodes_.push_back(BVHNode());

        buildRecursive(infos, begin, mid, maxLeafSize);
        int rightChild = (int)nodes_.size();
        buildRecursive(infos, mid, end, maxLeafSize);

        node.bounds = rangeAABB;
        node.objCount = 0;
        node.splitAxis = axis;
        node.firstObjOrChild = rightChild;

        nodes_[nodeIdx] = node;
    }
};

__device__ inline bool intersectSphereDevice(const Shape& s, const Ray& r, float& t, Vec3& n);
__device__ inline bool intersectTriangleDevice(const Shape& s, const Ray& r, float& t, Vec3& n);

__device__ inline bool intersectSceneFlat(
    const Shape* __restrict__ shapes,
    int shapeCount,
    const Ray& r,
    Hit& hit
) {
    bool found = false;
    hit.t = 1e20f;

    for (int i = 0; i < shapeCount; ++i) {
        float t;
        Vec3 n;
        bool ok = false;

        if (shapes[i].type == SHAPE_SPHERE)
            ok = intersectSphereDevice(shapes[i], r, t, n);
        else
            ok = intersectTriangleDevice(shapes[i], r, t, n);

        if (ok && t < hit.t) {
            hit.t = t;
            hit.id = i;
            hit.n = n;
            found = true;
        }
    }
    return found;
}

__device__ inline bool intersectBVH(
    const BVHNode* __restrict__ nodes,
    const Shape*   __restrict__ shapes,
    const Ray& r,
    Hit& hit
) {
    bool found = false;
    hit.t = 1e20f;
    Vec3 invDir(1.f / r.dir.x, 1.f / r.dir.y, 1.f / r.dir.z);

    int stack[64];
    int stackPtr = 0;
    int nodeIdx = 0;

    while (true) {
        const BVHNode& node = nodes[nodeIdx];

        if (!node.bounds.intersect(r, invDir, hit.t)) {
            if (stackPtr == 0) break;
            nodeIdx = stack[--stackPtr];
            continue;
        }

        if (node.objCount > 0) {
            // leaf: test all primitives
            for (int i = 0; i < node.objCount; ++i) {
                int shapeIdx = node.firstObjOrChild + i;
                const Shape& s = shapes[shapeIdx];

                float t;
                Vec3 n;
                bool ok = false;

                if (s.type == SHAPE_SPHERE)
                    ok = intersectSphereDevice(s, r, t, n);
                else
                    ok = intersectTriangleDevice(s, r, t, n);

                if (ok && t < hit.t) {
                    hit.t = t;
                    hit.id = shapeIdx;
                    hit.n = n;
                    found = true;
                }
            }

            if (stackPtr == 0) break;
            nodeIdx = stack[--stackPtr];
        } else {
            // interior: ordered traversal
            int leftIdx  = nodeIdx + 1;
            int rightIdx = node.firstObjOrChild;

            if (invDir[node.splitAxis] < 0.f) {
                int tmp = leftIdx; leftIdx = rightIdx; rightIdx = tmp;
            }

            nodeIdx = leftIdx;
            stack[stackPtr++] = rightIdx;
        }
    }

    return found;
}

__device__ inline bool intersectScene(
    const BVHNode* __restrict__ nodes,
    const Shape*   __restrict__ shapes,
    int shapeCount,
    const Ray& r,
    Hit& hit
) {
    const int BVH_THRESHOLD = 48;
    if (shapeCount <= BVH_THRESHOLD)
        return intersectSceneFlat(shapes, shapeCount, r, hit);
    else
        return intersectBVH(nodes, shapes, r, hit);
}


__device__ inline bool intersectSphereDevice(
    const Shape& s, const Ray& r, float& t, Vec3& n
) {
    const Sphere& sp = s.sphere;
    Vec3 op = r.orig - sp.center;
    float b = 2.f * op.dot(r.dir);
    float c = op.dot(op) - sp.radius * sp.radius;
    float discr = b * b - 4.f * c;
    if (discr < 0.f) return false;

    discr = sqrtf(discr);
    float t0 = (-b - discr) * 0.5f;
    float t1 = (-b + discr) * 0.5f;

    t = (t0 > EPS) ? t0 : ((t1 > EPS) ? t1 : 0.f);
    if (t <= 0.f) return false;

    Vec3 hp = r.orig + r.dir * t;
    n = (hp - sp.center).norm();
    
    return true;
}

__device__ inline bool intersectTriangleDevice(
    const Shape& s, const Ray& r, float& t, Vec3& n
) {
    const Triangle& tri = s.tri;
    Vec3 e1 = tri.v1 - tri.v0;
    Vec3 e2 = tri.v2 - tri.v0;
    Vec3 pvec = r.dir.cross(e2);
    float det = e1.dot(pvec);

    if (fabsf(det) < EPS) return false;
    float invDet = 1.f / det;

    Vec3 tvec = r.orig - tri.v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.f || u > 1.f) return false;

    Vec3 qvec = tvec.cross(e1);
    float v = r.dir.dot(qvec) * invDet;
    if (v < 0.f || u + v > 1.f) return false;

    t = e2.dot(qvec) * invDet;
    if (t <= EPS) return false;

    // interpolate vertex normals if available to achieve smooth shading
    float vnLenSq = tri.vn0.dot(tri.vn0);
    if (vnLenSq > 0.f) {
        float w = 1.f - u - v;
        n = (tri.vn0 * w + tri.vn1 * u + tri.vn2 * v).norm();
    } else {
        n = tri.normal;
    }

    return true;
}
