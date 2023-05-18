#ifndef SCENE_HPP
#define SCENE_HPP

#include "common.hpp"
#include "triangle.hpp"

/* Declarations */

struct Scene {
    Vector3 viewport;
    Vector3 cameraPosition, cameraRotation;
    Color backgroundColor;
    vector<Triangle> triangles;

    Vector3 toViewport(int x, int y, int canvasWidth, int canvasHeight);
    Color traceRay(Vector3 origin, Vector3 ray, double tMin, double tMax);

    static vector<Triangle> parseOBJ(string OBJPath);
    static Scene teapot();
};

/* Definitions */

Vector3 Scene::toViewport(int x, int y, int canvasWidth, int canvasHeight) {
    return {
        x * (this->viewport.x / canvasWidth), 
        y * (this->viewport.y / canvasHeight), 
        this->viewport.z
    };
}

Color Scene::traceRay(Vector3 origin, Vector3 ray, double tMin, double tMax) {
    double closestT = INFINITY;
    int closestTriangleIndex = -1;

    for (size_t i = 0; i < this->triangles.size(); i++) {
        tuple<bool, double> intersection = triangles[i].intersectRay(origin, ray);
        bool intersects = std::get<0>(intersection);
        double t = std::get<1>(intersection);

        if (intersects) {
            if (t > tMin && t < tMax && t < closestT) {
                closestT = t;
                closestTriangleIndex = static_cast<int>(i);
            }
        }
    }

    if (closestTriangleIndex == -1) {
        return backgroundColor;
    }

    return triangles[closestTriangleIndex].color;
}

vector<Triangle> Scene::parseOBJ(string OBJPath) {
    fstream fs(OBJPath, fstream::in);

    if (fs.fail()) {
        throw "Failed to open file.\n";
    }

    vector<Vector3> vertices;
    vector<Triangle> faces;

    while (!fs.eof()) {
        char id;
        double x, y, z;
        int i, j, k;

        fs >> id;

        if (id == 'v') {
            fs >> x >> y >> z;
            vertices.push_back({x, y, z});
        } else if (id == 'f') {
            fs >> i >> j >> k;
            faces.push_back(
                {
                    vertices[i - 1], 
                    vertices[j - 1], 
                    vertices[k - 1], 
                    {169, 169, 169}
                }
            );
        }
    }

    fs.close();

    return faces;
}

/* 6320 faces */
Scene Scene::teapot() {
    Scene scene = {
        .viewport = {1, 1, 1}, 
        .cameraPosition = {3, 2, -8}, 
        .cameraRotation = {0, -20, 0}, 
        .backgroundColor = {0, 0, 0}, 
        .triangles = parseOBJ("scenes/teapot.obj")
    };

    return scene;
}

#endif // SCENE_HPP
