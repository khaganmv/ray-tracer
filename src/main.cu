#include "scene.cuh"
#include "util.cuh"
#include <iostream>
#include <chrono>
#include <iomanip>

#define CANVAS_PATH "out/canvas.ppm"
#define CANVAS_WIDTH  1920
#define CANVAS_HEIGHT 1920

using namespace std::chrono;

/* Declarations */

__global__ 
void render(Color *canvas, Scene *scene);
void saveCanvas(Color *canvas);

/* Definitions */

int main() {
    Color *canvas;
    Scene *scene;
    SceneType sceneType = AURELIUS;

    /* Increase memory limit for recursion */
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, limit * 32);

    cudaMallocManaged(&canvas, CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(Color));
    cudaMallocManaged(&scene, sizeof(Scene));

    scene->initScene(sceneType);

    cudaPrefetch(canvas, CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(Color));
    cudaPrefetch(scene, sizeof(Scene));

    int tx = 8;
    int ty = 8;
    dim3 blocks(CANVAS_WIDTH / tx + 1, CANVAS_HEIGHT / ty + 1);
    dim3 threads(tx, ty);

    std::cout << std::fixed << std::setprecision(3);

    for (size_t n = 0; n < 30; n++) {
        auto bmStart = high_resolution_clock::now();

        render<<<blocks, threads>>>(canvas, scene);
        cudaDeviceSynchronize();

        auto bmStop = high_resolution_clock::now();
        auto bmDuration = duration_cast<milliseconds>(bmStop - bmStart);

        std::cout << static_cast<double>(bmDuration.count()) / 1000 << " ";
    }

    std::cout << "\n";

    try {
        saveCanvas(canvas);
    } catch (const char *e) {
        std::cerr << e;
        return -1;
    }

    cudaFree(canvas);
    cudaFree(scene->triangles);
    cudaFree(scene->directionalLights);
    cudaFree(scene->bvh.indices);
    cudaFree(scene->bvh.nodes);
    cudaFree(scene);

    return 0;
}

__global__ 
void render(Color *canvas, Scene *scene) {
    int x = threadIdx.x + blockIdx.x * blockDim.x - CANVAS_WIDTH / 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y - CANVAS_HEIGHT / 2;

    if (x >= CANVAS_WIDTH / 2 || y >= CANVAS_HEIGHT / 2) {
        return;
    }

    Vector3 ray = scene->toViewport(x, y, CANVAS_WIDTH, CANVAS_HEIGHT)
        .rotateX(scene->cameraRotation.x)
        .rotateY(scene->cameraRotation.y)
        .rotateZ(scene->cameraRotation.z);
    Color color = scene->traceRay(scene->cameraPosition, ray, 1.0, INFINITY, 3);
    
    int i = -y + CANVAS_HEIGHT / 2 - 1;
    int j = x + CANVAS_WIDTH / 2;
    canvas[i * CANVAS_HEIGHT + j] = color;
}

void saveCanvas(Color *canvas) {
    fstream fs(CANVAS_PATH, fstream::out | fstream::trunc | fstream::binary);

    if (fs.fail()) {
        throw "Failed to open file.\n";
    }

    fs << "P6\n" << CANVAS_WIDTH << " " << CANVAS_HEIGHT << " 255\n";

    for (int i = 0; i < CANVAS_HEIGHT; i++) {
        for (int j = 0; j < CANVAS_WIDTH; j++) {
            Color pixel = canvas[i * CANVAS_HEIGHT + j].normalize();

            fs << static_cast<unsigned char>(pixel.r) 
               << static_cast<unsigned char>(pixel.g) 
               << static_cast<unsigned char>(pixel.b);  
        }
    }

    fs.close();
}
