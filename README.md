# CUDA Path Tracer GM

## Overview

This project implements a basic path tracer using CUDA for GPU acceleration and a CPU fallback for comparison. It renders a 3D scene with a reflective metallic sphere, directional lighting, and random sampling for realistic global illumination effects.

The goal is to explore GPU programming with CUDA and develop a foundational ray/path tracing renderer with ray bounces, material shading, and gamma correction.

## Features

- Path tracing on GPU using CUDA with ray bouncing (up to `MAX_RAY_BOUNCES`)
- CPU-based ray tracing for performance and quality comparison
- Support for Lambertian (diffuse) and metallic materials
- Randomized anti-aliasing via multiple samples per pixel
- Gamma correction for color accuracy
- Outputs rendered image in PPM format

## Requirements

- Windows 10 or later
- Visual Studio 2022 (or compatible version) with C++ and CUDA support
- NVIDIA GPU with CUDA Toolkit installed (tested with CUDA 12.9)

## Building

1. Open the solution file `CudaGLViewer.sln` in Visual Studio.
2. Set the build configuration to `Release` and the platform to `x64`.
3. Build the solution (`Ctrl+Shift+B`).
4. Run program

Adjust parameters like image resolution, samples per pixel, and max ray depth inside the source code as needed.

## Notes

- The project currently renders a simple scene with one sphere with one direct "stage light-esque" light source and an ambient light source.
  
- Future improvements may include multiple objects, more material types, and real-time rendering integration (this was initially created with the idea of adding Unity compatability).

