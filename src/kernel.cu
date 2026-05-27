#include "kernel.cuh"
#include <curand_kernel.h>

constexpr float PI = 3.14159265358979323846f;
constexpr int MAX_BOUNCES = 100;

__device__ Vec3 sampleDiffuseHemisphere(const Vec3& nl, curandState* rng) {
    float r1 = 2.f * PI * curand_uniform(rng);
    float r2 = curand_uniform(rng);
    float r2s = sqrtf(r2);

    Vec3 u, v;
    create_orthonorm_sys(nl, u, v);

    return (u * cosf(r1) * r2s +
            v * sinf(r1) * r2s +
            nl * sqrtf(fmaxf(0.f, 1.f - r2))).norm();
}

__device__ Vec3 sampleSphereSolidAngle(
    const Vec3& hit_point,
    const Sphere& light,
    curandState* rng,
    float& omega_out
) {
    Vec3 sw = (light.center - hit_point).norm();
    Vec3 su, sv;
    create_orthonorm_sys(sw, su, sv);

    float d2 = (hit_point - light.center).dot(hit_point - light.center);
    float cos_a_max = sqrtf(fmaxf(0.f, 1.f - light.radius * light.radius / d2));
    omega_out = 2.f * PI * (1.f - cos_a_max);

    float u1 = curand_uniform(rng);
    float u2 = curand_uniform(rng);

    float cos_a = 1.f - u1 + u1 * cos_a_max;
    float sin_a = sqrtf(fmaxf(0.f, 1.f - cos_a * cos_a));
    float phi = 2.f * PI * u2;

    return (su * cosf(phi) * sin_a + sv * sinf(phi) * sin_a + sw * cos_a).norm();
}

extern "C" __global__
void renderKernel(
    Vec3* framebuffer,
    const BVHNode* bvhNodes,
    const Shape* shapes,
    int shapeCount,
    int width,
    int height,
    int spp
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (height - y - 1) * width + x;

    curandState rng;
    curand_init(1337ULL, idx, 0, &rng);

    Vec3 camOrig(50.f, 52.f, 295.6f);
    Vec3 camDir(0.f, -0.042612f, -1.f);
    camDir = camDir.norm();

    Vec3 camera_x(width * 0.5135f / height, 0.f, 0.f);
    Vec3 camera_y = camera_x.cross(camDir).norm() * 0.5135f;

    Vec3 pixel_col(0.f);

    for (int s = 0; s < spp; ++s) {
        float r1 = 2.f * curand_uniform(&rng);
        float dx = (r1 < 1.f) ? sqrtf(r1) - 1.f : 1.f - sqrtf(2.f - r1);

        float r2 = 2.f * curand_uniform(&rng);
        float dy = (r2 < 1.f) ? sqrtf(r2) - 1.f : 1.f - sqrtf(2.f - r2);

        float u = ((x + 0.5f + dx) / width) - 0.5f;
        float v = ((y + 0.5f + dy) / height) - 0.5f;

        Vec3 d = (camera_x * u + camera_y * v + camDir).norm();
        Ray ray(camOrig + d * 140.f, d);

        Vec3 radiance(0.f);
        Vec3 throughput(1.f);

        for (int depth = 0; depth < MAX_BOUNCES; ++depth) {
            Hit hit;
            if (!intersectScene(bvhNodes, shapes, shapeCount, ray, hit)) {
                break;
            }

            const Shape& obj = shapes[hit.id];
            Vec3 hit_point = ray.orig + ray.dir * hit.t;
            Vec3 n = hit.n;
            Vec3 nl = (n.dot(ray.dir) < 0.f) ? n : -n;

            if (obj.emis.x > 0.f || obj.emis.y > 0.f || obj.emis.z > 0.f) {
                radiance += throughput.mult(obj.emis);
                if (obj.color.x <= 0.f && obj.color.y <= 0.f && obj.color.z <= 0.f) {
                    break;
                }
            }

            float rr_prob = fmaxf(obj.color.x, fmaxf(obj.color.y, obj.color.z));
            if (depth > 5) {
                if (rr_prob <= 0.f || curand_uniform(&rng) >= rr_prob)
                    break;
                throughput /= rr_prob;
            }

            if (obj.refl == DIFF) {
                Vec3 direct(0.f);

                for (int i = 0; i < shapeCount; ++i) {
                    const Shape& lightObj = shapes[i];
                    if (lightObj.emis.x <= 0.f && lightObj.emis.y <= 0.f && lightObj.emis.z <= 0.f)
                        continue;
                    if (lightObj.type != SHAPE_SPHERE)
                        continue;

                    const Sphere& light = lightObj.sphere;
                    float omega = 0.f;
                    Vec3 samp_dir = sampleSphereSolidAngle(hit_point, light, &rng, omega);

                    Ray shadowRay(hit_point + nl * EPS, samp_dir);
                    Hit shadowHit;
                    if (intersectScene(bvhNodes, shapes, shapeCount, shadowRay, shadowHit) && shadowHit.id == i) {
                        float cosTheta = fmaxf(0.f, samp_dir.dot(nl));
                        direct += obj.color.mult(lightObj.emis) * (cosTheta * omega * (1.f / PI));
                    }
                }

                radiance += throughput.mult(direct);

                Vec3 ddir = sampleDiffuseHemisphere(nl, &rng);
                throughput = throughput.mult(obj.color);
                ray = Ray(hit_point + nl * EPS, ddir);
            }
            else if (obj.refl == SPEC) {
                Vec3 refl = ray.dir - n * (2.f * n.dot(ray.dir));
                throughput = throughput.mult(obj.color);
                ray = Ray(hit_point + refl * EPS, refl.norm());
            }
            else {
                Vec3 refl = ray.dir - n * (2.f * n.dot(ray.dir));
                bool into = n.dot(nl) > 0.f;

                float nc = 1.f;
                float nt = 1.5f;
                float nnt = into ? nc / nt : nt / nc;
                float ddn = ray.dir.dot(nl);
                float cos2t = 1.f - nnt * nnt * (1.f - ddn * ddn);

                throughput = throughput.mult(obj.color);

                if (cos2t < 0.f) {
                    ray = Ray(hit_point + refl * EPS, refl.norm());
                } else {
                    Vec3 tdir = (ray.dir * nnt - n * ((into ? 1.f : -1.f) * (ddn * nnt + sqrtf(cos2t)))).norm();

                    float a = nt - nc;
                    float b = nt + nc;
                    float c = 1.f - (into ? -ddn : tdir.dot(n));

                    float F0 = (a * a) / (b * b);
                    float Re = F0 + (1.f - F0) * c * c * c * c * c;
                    float Tr = 1.f - Re;

                    float prob = 0.25f + 0.5f * Re;
                    float refl_prob = Re / prob;
                    float trans_prob = Tr / (1.f - prob);

                    if (curand_uniform(&rng) < prob) {
                        throughput *= refl_prob;
                        ray = Ray(hit_point + refl * EPS, refl.norm());
                    } else {
                        throughput *= trans_prob;
                        ray = Ray(hit_point + tdir * EPS, tdir);
                    }
                }
            }
        }

        pixel_col += radiance;
    }

    framebuffer[idx] = pixel_col / float(spp);
}