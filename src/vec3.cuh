#pragma once

#include <cuda_runtime.h>
#include <math.h>

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3(float v = 0.f) : x(v), y(v), z(v) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ Vec3 operator*(float b) const { return Vec3(x * b, y * b, z * b); }
    __host__ __device__ Vec3 operator/(float b) const { return Vec3(x / b, y / b, z / b); }

    __host__ __device__ Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    __host__ __device__ Vec3& operator*=(float b) { x *= b; y *= b; z *= b; return *this; }
    __host__ __device__ Vec3& operator/=(float b) { x /= b; y /= b; z /= b; return *this; }

    __host__ __device__ float dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }

    __host__ __device__ Vec3 cross(const Vec3& b) const {
        return Vec3(
            y * b.z - z * b.y,
            z * b.x - x * b.z,
            x * b.y - y * b.x
        );
    }

    __host__ __device__ float length() const { return sqrtf(dot(*this)); }

    __host__ __device__ Vec3 norm() const {
        float len = length();
        return (len > 0.f) ? (*this / len) : Vec3(0.f);
    }

    __host__ __device__ Vec3 mult(const Vec3& b) const { return Vec3(x * b.x, y * b.y, z * b.z); }

    __host__ __device__ float operator[](int i) const { return (&x)[i]; }
    __host__ __device__ float& operator[](int i) { return (&x)[i]; }
};

__host__ __device__ inline Vec3 operator*(float a, const Vec3& b) { return b * a; }

__host__ __device__ inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}