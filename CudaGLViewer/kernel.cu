#include "common.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void initRandomStatesKernel(curandState* randStates, int width, int height, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    curand_init(seed + idx, 0, 0, &randStates[idx]);
}

__device__ Vec3 computeRayColorGpu(const Ray& ray, const Sphere& sphere, int depth, curandState* localRandState) {
    if (depth <= 0) return Vec3(0, 0, 0);

    float tHit;
    Vec3 hitPoint, normal;

    if (sphere.hit(ray, 0.001f, 10000.0f, tHit, hitPoint, normal)) {
        Ray scattered;
        Vec3 attenuation;
        bool didScatter = false;

        if (sphere.material.type == LAMBERTIAN) {
            didScatter = scatterLambertianGpu(ray, hitPoint, normal, attenuation, scattered, localRandState, sphere.material);
        }
        else if (sphere.material.type == METAL) {
            didScatter = scatterMetalGpu(ray, hitPoint, normal, attenuation, scattered, localRandState, sphere.material);
        }

        if (didScatter) {
            return attenuation * computeRayColorGpu(scattered, sphere, depth - 1, localRandState);
        }
        return Vec3(0, 0, 0);
    }

    // Background gradient (black to light blue)
    Vec3 unitDirection = ray.direction.normalize();
    float t = 0.5f * (unitDirection.y + 1.0f);
    return (1.0f - t) * Vec3(0, 0, 0) + t * Vec3(0.5f, 0.7f, 1.0f);
}

__global__ void renderKernel(Vec3* output, int width, int height, int samples, curandState* randStates, Sphere sphere) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curandState localRandState = randStates[idx];
    Vec3 pixelColor(0, 0, 0);

    // Camera parameters
    float aspectRatio = float(width) / float(height);
    float viewportHeight = 2.0f;
    float viewportWidth = aspectRatio * viewportHeight;
    float focalLength = 1.0f;

    Vec3 origin(0, 0, 0);
    Vec3 horizontal(viewportWidth, 0, 0);
    Vec3 vertical(0, viewportHeight, 0);
    Vec3 lowerLeftCorner = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, focalLength);

    for (int s = 0; s < samples; s++) {
        float u = (x + curand_uniform(&localRandState)) / float(width);
        float v = (y + curand_uniform(&localRandState)) / float(height);

        Ray r(origin, (lowerLeftCorner + u * horizontal + v * vertical - origin).normalize());
        pixelColor += computeRayColorGpu(r, sphere, MAX_RAY_BOUNCES, &localRandState);
    }

    pixelColor /= float(samples);

    // Clamp colors to [0, 1]
    pixelColor.x = fminf(fmaxf(pixelColor.x, 0.0f), 1.0f);
    pixelColor.y = fminf(fmaxf(pixelColor.y, 0.0f), 1.0f);
    pixelColor.z = fminf(fmaxf(pixelColor.z, 0.0f), 1.0f);

    // Gamma correction (gamma=2)
    pixelColor = Vec3(sqrtf(pixelColor.x), sqrtf(pixelColor.y), sqrtf(pixelColor.z));

    output[idx] = pixelColor;
    randStates[idx] = localRandState; // Save RNG state
}
