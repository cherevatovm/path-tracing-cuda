#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "bvh.cuh"
#include "kernel.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static inline int toInt(float x) {
    x = clampf(x, 0.f, 1.f);
    return int(powf(x, 1.f / 2.2f) * 255.f + 0.5f);
}

static Shape makeSphere(const Vec3& center, float radius, const Vec3& emis, const Vec3& color, ReflType refl) {
    Shape s{};
    s.type = SHAPE_SPHERE;
    s.sphere.center = center;
    s.sphere.radius = radius;
    s.emis = emis;
    s.color = color;
    s.refl = refl;
    return s;
}

static Shape makeTriangle(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& emis, const Vec3& color, ReflType refl) {
    Shape t{};
    t.type = SHAPE_TRIANGLE;
    t.tri.v0 = a;
    t.tri.v1 = b;
    t.tri.v2 = c;
    t.tri.normal = (b - a).cross(c - a).norm();
    t.emis = emis;
    t.color = color;
    t.refl = refl;
    return t;
}

static void addBox(std::vector<Shape>& shapes) {
    shapes.push_back(makeTriangle(Vec3(0, 0, 170), Vec3(0, 82.5f, 170), Vec3(0, 82.5f, 0), Vec3(0), Vec3(0.75f, 0.25f, 0.25f), DIFF));
    shapes.push_back(makeTriangle(Vec3(0, 0, 170), Vec3(0, 82.5f, 0), Vec3(0, 0, 0), Vec3(0), Vec3(0.75f, 0.25f, 0.25f), DIFF));

    shapes.push_back(makeTriangle(Vec3(99.5f, 0, 0), Vec3(99.5f, 82.5f, 0), Vec3(99.5f, 82.5f, 170), Vec3(0), Vec3(0.25f, 0.25f, 0.75f), DIFF));
    shapes.push_back(makeTriangle(Vec3(99.5f, 0, 0), Vec3(99.5f, 82.5f, 170), Vec3(99.5f, 0, 170), Vec3(0), Vec3(0.25f, 0.25f, 0.75f), DIFF));

    shapes.push_back(makeTriangle(Vec3(0, 0, 0), Vec3(0, 82.5f, 0), Vec3(99.5f, 82.5f, 0), Vec3(0), Vec3(0.25f, 0.75f, 0.25f), DIFF));
    shapes.push_back(makeTriangle(Vec3(0, 0, 0), Vec3(99.5f, 82.5f, 0), Vec3(99.5f, 0, 0), Vec3(0), Vec3(0.25f, 0.75f, 0.25f), DIFF));

    shapes.push_back(makeTriangle(Vec3(0, 0, 170), Vec3(0, 82.5f, 170), Vec3(99.5f, 82.5f, 170), Vec3(0), Vec3(0.75f, 0.75f, 0.75f), DIFF));
    shapes.push_back(makeTriangle(Vec3(0, 0, 170), Vec3(99.5f, 82.5f, 170), Vec3(99.5f, 0, 170), Vec3(0), Vec3(0.75f, 0.75f, 0.75f), DIFF));

    shapes.push_back(makeTriangle(Vec3(0, 0, 170), Vec3(99.5f, 0, 170), Vec3(99.5f, 0, 0), Vec3(0), Vec3(0.75f, 0.75f, 0.75f), DIFF));
    shapes.push_back(makeTriangle(Vec3(0, 0, 170), Vec3(99.5f, 0, 0), Vec3(0, 0, 0), Vec3(0), Vec3(0.75f, 0.75f, 0.75f), DIFF));

    shapes.push_back(makeTriangle(Vec3(0, 82.5f, 0), Vec3(99.5f, 82.5f, 0), Vec3(99.5f, 82.5f, 170), Vec3(0), Vec3(0.75f, 0.75f, 0.75f), DIFF));
    shapes.push_back(makeTriangle(Vec3(0, 82.5f, 0), Vec3(99.5f, 82.5f, 170), Vec3(0, 82.5f, 170), Vec3(0), Vec3(0.75f, 0.75f, 0.75f), DIFF));
}

static void addCube(std::vector<Shape>& shapes, const Vec3& c, float r, float alpha, const Vec3& emis, const Vec3& color, ReflType refl) {
    Vec3 p1{ c.x + r, c.y + r, c.z + r };
    Vec3 p7{ c.x - r, c.y - r, c.z - r };
    Vec3 p2{ p1.x, p1.y, p7.z };
    Vec3 p3{ p7.x, p1.y, p7.z };
    Vec3 p4{ p7.x, p1.y, p1.z };
    Vec3 p5{ p1.x, p7.y, p1.z };
    Vec3 p6{ p1.x, p7.y, p7.z };
    Vec3 p8{ p7.x, p7.y, p1.z };

    auto rot_y = [&](Vec3& p) {
        float xc = p.x - c.x;
        float zc = p.z - c.z;
        float nx = xc * cosf(alpha) - zc * sinf(alpha);
        float nz = xc * sinf(alpha) + zc * cosf(alpha);
        p.x = nx + c.x;
        p.z = nz + c.z;
    };

    rot_y(p1); rot_y(p2); rot_y(p3); rot_y(p4); rot_y(p5); rot_y(p6); rot_y(p7); rot_y(p8);

    shapes.push_back(makeTriangle(p1, p2, p3, emis, color, refl));
    shapes.push_back(makeTriangle(p3, p4, p1, emis, color, refl));
    shapes.push_back(makeTriangle(p8, p7, p6, emis, color, refl));
    shapes.push_back(makeTriangle(p6, p5, p8, emis, color, refl));
    shapes.push_back(makeTriangle(p2, p6, p7, emis, color, refl));
    shapes.push_back(makeTriangle(p7, p3, p2, emis, color, refl));
    shapes.push_back(makeTriangle(p1, p5, p6, emis, color, refl));
    shapes.push_back(makeTriangle(p6, p2, p1, emis, color, refl));
    shapes.push_back(makeTriangle(p8, p7, p3, emis, color, refl));
    shapes.push_back(makeTriangle(p3, p4, p8, emis, color, refl));
    shapes.push_back(makeTriangle(p1, p4, p8, emis, color, refl));
    shapes.push_back(makeTriangle(p8, p5, p1, emis, color, refl));
}

int main(int argc, char* argv[]) {
    int width = 1024;
    int height = 768;
    int spp = 128;

    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) spp = atoi(argv[3]);

    if (width <= 0 || height <= 0 || spp <= 0) {
        fprintf(stderr, "Usage: %s [width] [height] [spp]\n", argv[0]);
        fprintf(stderr, "  all arguments are optional positive integers\n");
        fprintf(stderr, "  default: %s 1024 768 128\n", argv[0]);
        return 1;
    }

    std::vector<Shape> shapes;
    addBox(shapes);

    shapes.push_back(makeSphere(Vec3(73, 16.5f, 95), 16.5f, Vec3(0), Vec3(1, 1, 1), REFR));
    addCube(shapes, Vec3(33, 15, 65), 15.f, float(M_PI) / 4.f, Vec3(0), Vec3(0.65f, 0.65f, 0.65f), SPEC);

    shapes.push_back(makeSphere(Vec3(50, 81.6f - 15.5f, 90), 5.5f, Vec3(40, 40, 40), Vec3(0), DIFF));

    std::vector<BVHNode> bvhNodes;
    BVHBuilder::build(shapes, bvhNodes, 4);

    printf("BVH built: %zu nodes for %zu primitives\n", bvhNodes.size(), shapes.size());

    Vec3* d_fb = nullptr;
    Shape* d_shapes = nullptr;
    BVHNode* d_bvhNodes = nullptr;

    CUDA_CHECK(cudaMalloc(&d_fb, width * height * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_shapes, shapes.size() * sizeof(Shape)));
    CUDA_CHECK(cudaMalloc(&d_bvhNodes, bvhNodes.size() * sizeof(BVHNode)));

    CUDA_CHECK(cudaMemcpy(d_shapes, shapes.data(), shapes.size() * sizeof(Shape), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bvhNodes, bvhNodes.data(), bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));

    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    renderKernel<<<grid, block>>>(d_fb, d_bvhNodes, d_shapes, (int)shapes.size(), width, height, spp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Vec3> framebuffer(width * height);
    CUDA_CHECK(cudaMemcpy(framebuffer.data(), d_fb, width * height * sizeof(Vec3), cudaMemcpyDeviceToHost));

    std::vector<unsigned char> img(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        img[i * 3 + 0] = (unsigned char)toInt(framebuffer[i].x);
        img[i * 3 + 1] = (unsigned char)toInt(framebuffer[i].y);
        img[i * 3 + 2] = (unsigned char)toInt(framebuffer[i].z);
    }

    stbi_write_png("result.png", width, height, 3, img.data(), width * 3);

    CUDA_CHECK(cudaFree(d_fb));
    CUDA_CHECK(cudaFree(d_shapes));
    CUDA_CHECK(cudaFree(d_bvhNodes));

    return 0;
}