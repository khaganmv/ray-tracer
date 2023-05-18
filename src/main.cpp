#include "scene.hpp"
#include <iostream>

#define CANVAS_PATH "out/canvas.ppm"
#define CANVAS_WIDTH  600
#define CANVAS_HEIGHT 600

/* Declarations */

array<array<Color, CANVAS_WIDTH>, CANVAS_HEIGHT> canvas;

void putPixel(int x, int y, Color color);
void saveCanvas();

/* Definitions */

int main() {
    Scene scene;

    try {
        scene = Scene::teapot();
    } catch (const char *e) {
        std::cerr << e;
        return -1;
    }

    for (int x = -CANVAS_WIDTH / 2; x < CANVAS_WIDTH / 2; x++) {
        for (int y = -CANVAS_HEIGHT / 2; y < CANVAS_HEIGHT / 2; y++) {
            Vector3 ray = scene
                .toViewport(x, y, CANVAS_WIDTH, CANVAS_HEIGHT)
                .rotateX(scene.cameraRotation.x)
                .rotateY(scene.cameraRotation.y)
                .rotateZ(scene.cameraRotation.z);
            Color color = scene.traceRay(scene.cameraPosition, ray, 1.0, INFINITY);
            
            putPixel(x, y, color);
        }
    }

    try {
        saveCanvas();
    } catch (const char *e) {
        std::cerr << e;
        return -1;
    }

    return 0;
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
