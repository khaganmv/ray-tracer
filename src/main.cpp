#include "scene.hpp"
#include <iostream>
#include <thread>
#include <chrono>

#define CANVAS_PATH "out/canvas.ppm"
#define CANVAS_WIDTH  512
#define CANVAS_HEIGHT 512

using std::thread;
using namespace std::chrono;

/* Declarations */

array<array<Color, CANVAS_WIDTH>, CANVAS_HEIGHT> canvas;

void render(Scene scene, int xMin, int xMax);
void putPixel(int x, int y, Color color);
void saveCanvas();

/* Definitions */

int main() {
    Scene scene;

    try {
        scene = Scene::serapis();
    } catch (const char *e) {
        std::cerr << e;
        return -1;
    }

    vector<thread> threads;
    int threadsSize = 8;

    int start = -CANVAS_WIDTH / 2;
    int step = CANVAS_WIDTH / threadsSize;

    auto bmStart = high_resolution_clock::now();

    for (int i = 0; i < threadsSize; i++) {
        threads.push_back(thread(render, scene, start, start + step));
        start += step;
    }

    for (int i = 0; i < threadsSize; i++) {
        threads[i].join();
    }

    auto bmStop = high_resolution_clock::now();
    auto bmDuration = duration_cast<milliseconds>(bmStop - bmStart);

    std::cout << "Duration: " 
              << static_cast<double>(bmDuration.count()) / 1000 
              << " seconds.\n";

    try {
        saveCanvas();
    } catch (const char *e) {
        std::cerr << e;
        return -1;
    }

    return 0;
}

void render(Scene scene, int xMin, int xMax) {
    for (int x = xMin; x < xMax; x++) {
        for (int y = -CANVAS_HEIGHT / 2; y < CANVAS_HEIGHT / 2; y++) {
            Vector3 ray = scene
                .toViewport(x, y, CANVAS_WIDTH, CANVAS_HEIGHT)
                .rotateX(scene.cameraRotation.x)
                .rotateY(scene.cameraRotation.y)
                .rotateZ(scene.cameraRotation.z);
            Color color = scene.traceRay(scene.cameraPosition, ray, 1.0, INFINITY, 3);
            
            putPixel(x, y, color);
        }
    }
}

void putPixel(int x, int y, Color color) {
    int i = -y + CANVAS_HEIGHT / 2 - 1;
    int j = x + CANVAS_WIDTH / 2;
    canvas[i][j] = color;
}

void saveCanvas() {
    fstream fs(CANVAS_PATH, fstream::out | fstream::trunc | fstream::binary);

    if (fs.fail()) {
        throw "Failed to open file.\n";
    }

    fs << "P6\n" << CANVAS_WIDTH << " " << CANVAS_HEIGHT << " 255\n";

    for (int i = 0; i < CANVAS_HEIGHT; i++) {
        for (int j = 0; j < CANVAS_WIDTH; j++) {
            Color pixel = canvas[i][j].normalize();

            fs << static_cast<unsigned char>(pixel.r) 
               << static_cast<unsigned char>(pixel.g) 
               << static_cast<unsigned char>(pixel.b);  
        }
    }

    fs.close();
}
