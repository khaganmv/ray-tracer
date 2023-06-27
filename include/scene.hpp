#ifndef SCENE_HPP
#define SCENE_HPP

#include "common.hpp"
#include "triangle.hpp"
#include "light.hpp"
#include "bvh.hpp"
#include <iostream>
#include <chrono>

using namespace std::chrono;

/* Declarations */

struct Scene {
    Vector3 viewport;
    Vector3 cameraPosition, cameraRotation;
    Color backgroundColor;
    vector<Triangle> triangles;
    double ambientLight;
    vector<Point> pointLights;
    vector<Directional> directionalLights;
    BVH bvh;

    Scene() = default;
    Scene(
        Vector3 _viewport, 
        Vector3 _cameraPosition, Vector3 _cameraRotation, 
        Color _backgroundColor, 
        vector<Triangle> _triangles, 
        double _ambientLight, 
        vector<Point> _pointLights, 
        vector<Directional> _directionalLights
    ) : viewport(_viewport), 
        cameraPosition(_cameraPosition), cameraRotation(_cameraRotation), 
        backgroundColor(_backgroundColor), 
        triangles(_triangles), 
        ambientLight(_ambientLight), 
        pointLights(_pointLights), 
        directionalLights(_directionalLights) {
#if USE_BVH
        auto bmStart = high_resolution_clock::now();

        bvh = BVH(triangles);

        auto bmStop = high_resolution_clock::now();
        auto bmDuration = duration_cast<milliseconds>(bmStop - bmStart);

        std::cout << "[ BVH ] " 
                  << static_cast<double>(bmDuration.count()) / 1000 
                  << " seconds.\n";
#endif
    }

    Vector3 toViewport(int x, int y, int canvasWidth, int canvasHeight);
    Color traceRay(
        Vector3 origin, Vector3 ray, double tMin, double tMax, int recursionDepth
    );
    double computeTotalLighting(
        Vector3 point, Vector3 normal, Vector3 inverse, int specularity
    );
    double computeLighting(
        double intensity, Vector3 light, double tMax, 
        Vector3 point, Vector3 normal, Vector3 inverse, int specularity
    );
    Vector3 reflectRay(Vector3 ray, Vector3 normal);
    tuple<bool, double, int> closestIntersection(
        Vector3 origin, Vector3 ray, double tMin, double tMax
    );

    static vector<Triangle> parseOBJ(string OBJPath);
    static Scene teapot();
    static Scene bunny();
    static Scene erato();
    static Scene dragon();
    static Scene aurelius();
};

/* Definitions */

Vector3 Scene::toViewport(int x, int y, int canvasWidth, int canvasHeight) {
    return {
        x * (this->viewport.x / canvasWidth), 
        y * (this->viewport.y / canvasHeight), 
        this->viewport.z
    };
}

Color Scene::traceRay(
    Vector3 origin, Vector3 ray, double tMin, double tMax, int recursionDepth
) {
#if USE_BVH
    tuple<bool, double, int> intersection = bvh.intersectRay(origin, ray, tMin, tMax, 0);
#else
    tuple<bool, double, int> intersection = closestIntersection(origin, ray, tMin, tMax);
#endif
    bool intersectsAny = std::get<0>(intersection);
    double closestT = std::get<1>(intersection);
    int closestTriangleIndex = std::get<2>(intersection);

    if (!intersectsAny) {
        return backgroundColor;
    }

    Vector3 point = origin + closestT * ray;

    Triangle closestTriangle = triangles[closestTriangleIndex];
    Color color = closestTriangle.color;
    Vector3 normal = closestTriangle.normal;
    int specularity = closestTriangle.specularity;
    double reflectivity = closestTriangle.reflectivity;

    Color localColor = color * computeTotalLighting(point, normal, -ray, specularity);

    if (recursionDepth == 0 || reflectivity < 0) {
        return localColor;
    }

    Vector3 reflected = reflectRay(-ray, normal);
    Color reflectedColor = traceRay(point, reflected, 0.001, INFINITY, recursionDepth - 1);
    
    return localColor * (1 - reflectivity) + reflectedColor * reflectivity;
}

double Scene::computeTotalLighting(
    Vector3 point, Vector3 normal, Vector3 inverse, int specularity
) {
    double totalIntensity = ambientLight;

    for (Point pointLight : pointLights) {
        totalIntensity += computeLighting(
            pointLight.intensity, pointLight.position - point, 1, 
            point, normal, inverse, specularity
        );
    }

    for (Directional directionalLight : directionalLights) {
        totalIntensity += computeLighting(
            directionalLight.intensity, directionalLight.direction, INFINITY, 
            point, normal, inverse, specularity
        );
    }

    return totalIntensity;
}

double Scene::computeLighting(
    double intensity, Vector3 light, double tMax, 
    Vector3 point, Vector3 normal, Vector3 inverse, int specularity
) {
    double totalIntensity = 0.0;

    /* Shadow check */
#if USE_BVH
    tuple<bool, double, int> intersection = bvh.intersectRay(point, light, 0.001, tMax, 0);
#else
    tuple<bool, double, int> intersection = closestIntersection(point, light, 0.001, tMax);
#endif
    bool intersectsAny = std::get<0>(intersection);

    if (intersectsAny) {
        return totalIntensity;
    }

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

tuple<bool, double, int> Scene::closestIntersection(
    Vector3 origin, Vector3 ray, double tMin, double tMax
) {
    bool intersectsAny = false;
    double closestT = INFINITY;
    int closestTriangleIndex = -1;

    for (size_t i = 0; i < this->triangles.size(); i++) {
        tuple<bool, double> intersection = triangles[i].intersectRay(origin, ray);
        bool intersects = std::get<0>(intersection);
        double t = std::get<1>(intersection);

        if (intersects) {
            if (t > tMin && t < tMax && t < closestT) {
                intersectsAny = true;
                closestT = t;
                closestTriangleIndex = static_cast<int>(i);
            }
        }
    }

    return { intersectsAny, closestT, closestTriangleIndex };
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
                    {255, 255, 255}, 
                    1, 
                    -1
                }
            );
        }
    }

    std::cout << "[ TRI ] " << faces.size() << " triangles.\n";

    fs.close();

    return faces;
}

/* 6330 faces */
Scene Scene::teapot() {
    Scene scene = {
        {1, 1, 1}, 
        {-0.015, 4, -11.99}, 
        {0, 0.1, 0}, 
        {255, 255, 255}, 
        parseOBJ("scenes/teapot.obj"), 
        0.2, 
        {}, 
        {
            {
                0.5, 
                {0, 1, -1}
            }
        }
    };

    for (size_t i = 0; i < scene.triangles.size() - 10; i++) {
        scene.triangles[i].reflectivity = 0.2;
    }

    for (size_t i = scene.triangles.size() - 10; i < scene.triangles.size() - 4; i++) {
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 4; i < scene.triangles.size() - 2; i++) {
        scene.triangles[i].color = {0, 255, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 2; i < scene.triangles.size(); i++) {
        scene.triangles[i].color = {255, 0, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    return scene;
}

/* 144056 faces */
Scene Scene::bunny() {
    Scene scene = {
        {1, 1, 1}, 
        {0.1425, 2, -5.94}, 
        {0, 0.1, 0}, 
        {255, 255, 255}, 
        parseOBJ("scenes/bunny.obj"), 
        0.2, 
        {}, 
        {
            {
                0.5, 
                {0, 1, -1}
            }
        }
    };

    for (size_t i = 0; i < scene.triangles.size() - 10; i++) {
        scene.triangles[i].reflectivity = 0.2;
    }

    for (size_t i = scene.triangles.size() - 10; i < scene.triangles.size() - 4; i++) {
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 4; i < scene.triangles.size() - 2; i++) {
        scene.triangles[i].color = {0, 255, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 2; i < scene.triangles.size(); i++) {
        scene.triangles[i].color = {255, 0, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    return scene;
}

/* 412508 faces */
Scene Scene::erato() {
    Scene scene = {
        {1, 1, 1}, 
        {-0.8, 28.7, -86.1}, 
        {0, 0.1, 0}, 
        {255, 255, 255}, 
        parseOBJ("scenes/erato.obj"), 
        0.2, 
        {}, 
        {
            {
                0.5, 
                {0, 1, -1}
            }
        }
    };

    for (size_t i = 0; i < scene.triangles.size() - 10; i++) {
        scene.triangles[i].reflectivity = 0.2;
    }

    for (size_t i = scene.triangles.size() - 10; i < scene.triangles.size() - 4; i++) {
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 4; i < scene.triangles.size() - 2; i++) {
        scene.triangles[i].color = {0, 255, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 2; i < scene.triangles.size(); i++) {
        scene.triangles[i].color = {255, 0, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    return scene;
}

/* 871316 faces */
Scene Scene::dragon() {
    Scene scene = {
        {1, 1, 1}, 
        {-0.0425, 0.7115, -3.01725}, 
        {0, 0.1, 0}, 
        {255, 255, 255}, 
        parseOBJ("scenes/dragon.obj"), 
        0.2, 
        {}, 
        {
            {
                0.5, 
                {0, 1, -1}
            }
        }
    };

    for (size_t i = 0; i < scene.triangles.size() - 10; i++) {
        scene.triangles[i].reflectivity = 0.2;
    }

    for (size_t i = scene.triangles.size() - 10; i < scene.triangles.size() - 4; i++) {
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 4; i < scene.triangles.size() - 2; i++) {
        scene.triangles[i].color = {0, 255, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 2; i < scene.triangles.size(); i++) {
        scene.triangles[i].color = {255, 0, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    return scene;
}

/* 1704768 faces */
Scene Scene::aurelius() {
    Scene scene = {
        {1, 1, 1}, 
        {-0.025, 3.05, -20.9}, 
        {0, 0.1, 0}, 
        {255, 255, 255}, 
        parseOBJ("scenes/aurelius.obj"), 
        0.2, 
        {}, 
        {
            {
                0.5, 
                {0, 1, -1}
            }
        }
    };

    for (size_t i = 0; i < scene.triangles.size() - 10; i++) {
        scene.triangles[i].reflectivity = 0.2;
    }

    for (size_t i = scene.triangles.size() - 10; i < scene.triangles.size() - 4; i++) {
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 4; i < scene.triangles.size() - 2; i++) {
        scene.triangles[i].color = {0, 255, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    for (size_t i = scene.triangles.size() - 2; i < scene.triangles.size(); i++) {
        scene.triangles[i].color = {255, 0, 0};
        scene.triangles[i].reflectivity = 0.4;
    }

    return scene;
}

#endif // SCENE_HPP
