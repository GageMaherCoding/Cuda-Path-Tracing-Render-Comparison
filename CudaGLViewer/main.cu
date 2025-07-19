#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "common.h"
#include <fstream>

#define WIDTH 800
#define HEIGHT 600
#define SAMPLES 50

// Kernel declarations
__global__ void initRandomStatesKernel(curandState* randStates, int width, int height, unsigned int seed);

__global__ void renderKernel(Vec3* output, int width, int height, int samples, curandState* randStates, Sphere sph);

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// CPU rayColor function — only one CPU function not in common.h
Vec3 rayColorCpu(const Ray& r, const Sphere& sph, int depth) {
    if (depth <= 0) return Vec3(0, 0, 0);

    float tHit;
    Vec3 hitPoint, normal;
    if (sph.hit(r, 0.001f, 10000.0f, tHit, hitPoint, normal)) {
        Ray scattered;
        Vec3 attenuation;
        bool scatteredOk = false;

        if (sph.material.type == LAMBERTIAN) {
            scatteredOk = scatterLambertianCpu(r, hitPoint, normal, attenuation, scattered, sph.material);
        }
        else if (sph.material.type == METAL) {
            scatteredOk = scatterMetalCpu(r, hitPoint, normal, attenuation, scattered, sph.material);
        }


        if (scatteredOk) {
            return attenuation * rayColorCpu(scattered, sph, depth - 1);
        }
        return Vec3(0, 0, 0);
    }

    Vec3 unitDir = r.direction.normalize();
    float t = 0.5f * (unitDir.y + 1.0f);
    return (1.0f - t) * Vec3(0.0f, 0.0f, 0.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
}

int main() {
    int imgSize = WIDTH * HEIGHT;
    size_t imgBytes = imgSize * sizeof(Vec3);

    Vec3* hImageGpu = new Vec3[imgSize];
    Vec3* hImageCpu = new Vec3[imgSize];

    Vec3* dImage = nullptr;
    curandState* dRandStates = nullptr;
    checkCudaErrors(cudaMalloc(&dImage, imgBytes));
    checkCudaErrors(cudaMalloc(&dRandStates, imgSize * sizeof(curandState)));

    Material mat{ METAL, Vec3(0.9f, 0.9f, 0.9f), 0.1f };
    Sphere hSphere(Vec3(0, 0, -1), 0.5f, mat);

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    initRandomStatesKernel << <grid, block >> > (dRandStates, WIDTH, HEIGHT, time(0));

    checkCudaErrors(cudaDeviceSynchronize());

    auto gpuStart = std::chrono::high_resolution_clock::now();
    renderKernel << <grid, block >> > (dImage, WIDTH, HEIGHT, SAMPLES, dRandStates, hSphere);
    checkCudaErrors(cudaDeviceSynchronize());
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    double gpuDuration = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();

    checkCudaErrors(cudaMemcpy(hImageGpu, dImage, imgBytes, cudaMemcpyDeviceToHost));
    std::cout << "GPU Render Time: " << gpuDuration << " ms" << std::endl;

    {
        std::ofstream ofs("render_gpu.ppm");
        ofs << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
        for (int y = HEIGHT - 1; y >= 0; y--) {
            for (int x = 0; x < WIDTH; x++) {
                Vec3 color = hImageGpu[y * WIDTH + x];
                color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
                color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
                color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);

                int ir = static_cast<int>(255.99f * color.x);
                int ig = static_cast<int>(255.99f * color.y);
                int ib = static_cast<int>(255.99f * color.z);
                ofs << ir << " " << ig << " " << ib << "\n";
            }
        }
        ofs.close();
        std::cout << "Saved GPU render to render_gpu.ppm" << std::endl;
    }

    auto cpuStart = std::chrono::high_resolution_clock::now();
    float aspectRatio = float(WIDTH) / float(HEIGHT);
    float viewportHeight = 2.0f;
    float viewportWidth = aspectRatio * viewportHeight;
    float focalLength = 1.0f;

    Vec3 origin(0, 0, 0);
    Vec3 horizontal(viewportWidth, 0, 0);
    Vec3 vertical(0, viewportHeight, 0);
    Vec3 lowerLeftCorner = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, focalLength);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            Vec3 color(0, 0, 0);
            for (int s = 0; s < SAMPLES; s++) {
                float u = (x + (rand() / (float)RAND_MAX)) / float(WIDTH);
                float v = (y + (rand() / (float)RAND_MAX)) / float(HEIGHT);
                Ray r(origin, (lowerLeftCorner + u * horizontal + v * vertical - origin).normalize());
                color += rayColorCpu(r, hSphere, MAX_RAY_BOUNCES);
            }
            color /= float(SAMPLES);
            color = Vec3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z));

            color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
            color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
            color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);

            hImageCpu[y * WIDTH + x] = color;
        }
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuDuration = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "CPU Render Time: " << cpuDuration << " ms" << std::endl;

    {
        std::ofstream ofs("render_cpu.ppm");
        ofs << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
        for (int y = HEIGHT - 1; y >= 0; y--) {
            for (int x = 0; x < WIDTH; x++) {
                Vec3 color = hImageCpu[y * WIDTH + x];
                color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
                color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
                color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);

                int ir = static_cast<int>(255.99f * color.x);
                int ig = static_cast<int>(255.99f * color.y);
                int ib = static_cast<int>(255.99f * color.z);
                ofs << ir << " " << ig << " " << ib << "\n";
            }
        }
        ofs.close();
        std::cout << "Saved CPU render to render_cpu.ppm" << std::endl;
    }

    cudaFree(dImage);
    cudaFree(dRandStates);
    delete[] hImageGpu;
    delete[] hImageCpu;

    return 0;
}