#ifndef SCENE_HPP
#define SCENE_HPP

#include "common.hpp"
#include "triangle.hpp"
#include "light.hpp"

/* Declarations */

struct Scene {
    Vector3 viewport;
    Vector3 cameraPosition, cameraRotation;
    Color backgroundColor;
    vector<Triangle> triangles;
    double ambientLight;
    vector<Point> pointLights;
    vector<Directional> directionalLights;

    Vector3 toViewport(int x, int y, int canvasWidth, int canvasHeight);
    Color traceRay(Vector3 origin, Vector3 ray, double tMin, double tMax);
    double computeTotalLighting(
        Vector3 point, Vector3 normal, Vector3 inverse, int specularity
    );
    double computeLighting(
        double intensity, Vector3 light, 
        Vector3 normal, Vector3 inverse, int specularity
    );
    Vector3 reflectRay(Vector3 ray, Vector3 normal);

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

    Vector3 point = origin + closestT * ray;

    Triangle closestTriangle = triangles[closestTriangleIndex];
    Color color = closestTriangle.color;
    Vector3 normal = closestTriangle.normal;
    int specularity = closestTriangle.specularity;

    return color * computeTotalLighting(point, normal, -ray, specularity);
}

double Scene::computeTotalLighting(
    Vector3 point, Vector3 normal, Vector3 inverse, int specularity
) {
    double totalIntensity = ambientLight;

    for (Point pointLight : pointLights) {
        totalIntensity += computeLighting(
            pointLight.intensity, pointLight.position - point, 
            normal, inverse, specularity
        );
    }

    for (Directional directionalLight : directionalLights) {
        totalIntensity += computeLighting(
            directionalLight.intensity, directionalLight.direction, 
            normal, inverse, specularity
        );
    }

    return totalIntensity;
}

double Scene::computeLighting(
    double intensity, Vector3 light, 
    Vector3 normal, Vector3 inverse, int specularity
) {
    double totalIntensity = 0.0;
    double normalDotLight = normal.dot(light); // ||N|| * ||L|| * cos(a)

    if (normalDotLight > 0.0) {
        double lightCos = normalDotLight / (normal.magnitude() * light.magnitude());
        totalIntensity += intensity * lightCos;
    }

    if (specularity != -1) {
        Vector3 reflected = reflectRay(light, normal);
        double reflectedDotInverse = reflected.dot(inverse); // ||R|| * ||V|| * cos(a)

        if (reflectedDotInverse > 0) {
            double reflectedCos = reflectedDotInverse / (reflected.magnitude() * inverse.magnitude());
            totalIntensity += intensity * pow(reflectedCos, specularity);
        }
    }

    return totalIntensity;
}

Vector3 Scene::reflectRay(Vector3 ray, Vector3 normal) {
    return 2 * normal * normal.dot(ray) - ray;
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
                    {169, 169, 169}, 
                    1
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
        .triangles = parseOBJ("scenes/teapot.obj"), 
        .ambientLight = 0.2, 
        .pointLights = {}, 
        .directionalLights = {
            {
                0.6, 
                {-1, 0, -1}
            }
        }
    };

    return scene;
}

#endif // SCENE_HPP
