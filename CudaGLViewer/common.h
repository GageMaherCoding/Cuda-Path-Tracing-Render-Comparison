#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// ==================== Vec3 ====================
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float val) : x(val), y(val), z(val) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(float t) const { return Vec3(x * t, y * t, z * t); }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ Vec3 operator/(float t) const { float inv = 1.0f / t; return Vec3(x * inv, y * inv, z * inv); }
    __host__ __device__ Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__ Vec3& operator/=(float t) { float inv = 1.0f / t; x *= inv; y *= inv; z *= inv; return *this; }

    __host__ __device__ bool nearZero() const {
        const float s = 1e-6f;
        return (fabsf(x) < s) && (fabsf(y) < s) && (fabsf(z) < s);
    }

    __host__ __device__ float lengthSquared() const { return x * x + y * y + z * z; }
    __host__ __device__ float length() const { return sqrtf(lengthSquared()); }
    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return (len > 0) ? (*this / len) : Vec3(0, 0, 0);
    }
};

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) { return v * t; }
__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - n * 2.0f * dot(v, n);
}

// Optional color clamp for CPU output
inline Vec3 clampColor(const Vec3& c) {
    return Vec3(
        fminf(fmaxf(c.x, 0.0f), 1.0f),
        fminf(fmaxf(c.y, 0.0f), 1.0f),
        fminf(fmaxf(c.z, 0.0f), 1.0f)
    );
}

// ==================== Ray ====================
struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}
    __host__ __device__ Vec3 at(float t) const { return origin + direction * t; }
};

// ==================== Material ====================
enum MaterialType { LAMBERTIAN, METAL };

struct Material {
    MaterialType type;
    Vec3 albedo;
    float fuzz;  // For metal only, 0-1
};

// ==================== Sphere ====================
struct Sphere {
    Vec3 center;
    float radius;
    Material material;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(Vec3 c, float r, Material m) : center(c), radius(r), material(m) {}

    __host__ __device__ bool hit(const Ray& r, float tMin, float tMax, float& tHit, Vec3& hitPoint, Vec3& normal) const {
        Vec3 oc = r.origin - center;
        float a = dot(r.direction, r.direction);
        float b = dot(oc, r.direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float sqrtD = sqrtf(discriminant);
            float root = (-b - sqrtD) / a;
            if (root < tMax && root > tMin) {
                tHit = root;
                hitPoint = r.at(tHit);
                normal = (hitPoint - center) / radius;
                return true;
            }
            root = (-b + sqrtD) / a;
            if (root < tMax && root > tMin) {
                tHit = root;
                hitPoint = r.at(tHit);
                normal = (hitPoint - center) / radius;
                return true;
            }
        }
        return false;
    }
};

// ==================== Random Helpers (CPU) ====================
inline Vec3 randomInUnitSphereCpu() {
    while (true) {
        Vec3 p(
            2.0f * (rand() / (float)RAND_MAX) - 1.0f,
            2.0f * (rand() / (float)RAND_MAX) - 1.0f,
            2.0f * (rand() / (float)RAND_MAX) - 1.0f
        );
        if (p.length() < 1.0f) return p;
    }
}
inline Vec3 randomUnitVectorCpu() {
    return randomInUnitSphereCpu().normalize();
}

// ==================== Scatter (CPU) ====================
inline bool scatterLambertianCpu(const Ray& rIn, const Vec3& hitPoint, const Vec3& normal, Vec3& attenuation, Ray& scattered, const Material& material) {
    Vec3 lightDir = Vec3(1, 1, -1).normalize();
    float ambient = 0.1f;
    float spotlight = fmaxf(0.0f, dot(normal, lightDir)) * 8.0f;
    attenuation = material.albedo * (ambient + spotlight);

    Vec3 scatterDir = normal + randomUnitVectorCpu();
    float len = scatterDir.length();
    if (len < 1e-6f) scatterDir = normal;
    else scatterDir /= len;
    scattered = Ray(hitPoint, scatterDir);

    return true;
}

inline bool scatterMetalCpu(const Ray& rIn, const Vec3& hitPoint, const Vec3& normal, Vec3& attenuation, Ray& scattered, const Material& material) {
    Vec3 lightDir = Vec3(1, 1, -1).normalize();
    float ambient = 0.1f;
    float diffuse = fmaxf(dot(normal, lightDir), 0.0f) * 8.0f;
    float spec = powf(fmaxf(dot(reflect(-lightDir, normal), rIn.direction.normalize()), 0.0f), 64) * 8.0f;
    Vec3 reflected = reflect(rIn.direction.normalize(), normal);
    Vec3 fuzzVec = randomInUnitSphereCpu() * material.fuzz;
    scattered = Ray(hitPoint, (reflected + fuzzVec).normalize());

    attenuation = material.albedo * (ambient + diffuse) + Vec3(1.0f) * spec;

    return (dot(scattered.direction, normal) > 0);
}

// ==================== Random Helpers (GPU) ====================
__device__ __inline__ Vec3 randomInUnitSphere(curandState* localRandState) {
    Vec3 p;
    do {
        p = 2.0f * Vec3(
            curand_uniform(localRandState),
            curand_uniform(localRandState),
            curand_uniform(localRandState)
        ) - Vec3(1, 1, 1);
    } while (p.length() >= 1.0f);
    return p;
}

__device__ __inline__ Vec3 randomUnitVector(curandState* localRandState) {
    return randomInUnitSphere(localRandState).normalize();
}

// ==================== Scatter (GPU) ====================
__device__ __inline__ bool scatterLambertianGpu(
    const Ray& rIn, const Vec3& hitPoint, const Vec3& normal,
    Vec3& attenuation, Ray& scattered,
    curandState* localRandState, const Material& material
) {
    Vec3 lightDir = Vec3(1, 1, -1).normalize();
    float ambient = 0.1f;
    float spotlight = fmaxf(0.0f, dot(normal, lightDir)) * 8.0f;
    attenuation = material.albedo * (ambient + spotlight);

    Vec3 scatterDir = normal + randomUnitVector(localRandState);
    float len = scatterDir.length();
    if (len < 1e-6f) scatterDir = normal;
    else scatterDir /= len;

    scattered = Ray(hitPoint, scatterDir);
    return true;
}

__device__ __inline__ bool scatterMetalGpu(
    const Ray& rIn, const Vec3& hitPoint, const Vec3& normal,
    Vec3& attenuation, Ray& scattered,
    curandState* localRandState, const Material& material
) {
    Vec3 lightDir = Vec3(1, 1, -1).normalize();
    float ambient = 0.1f;
    float diffuse = fmaxf(dot(normal, lightDir), 0.0f) * 8.0f;
    float spec = powf(fmaxf(dot(reflect(-lightDir, normal), rIn.direction.normalize()), 0.0f), 64) * 8.0f;
    Vec3 reflected = reflect(rIn.direction.normalize(), normal);
    Vec3 fuzzVec = randomInUnitSphere(localRandState) * material.fuzz;
    scattered = Ray(hitPoint, (reflected + fuzzVec).normalize());

    attenuation = material.albedo * (ambient + diffuse) + Vec3(1.0f) * spec;

    return (dot(scattered.direction, normal) > 0);
}

#define MAX_RAY_BOUNCES 10
