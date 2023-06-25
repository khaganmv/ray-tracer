#ifndef SCENE_CUH
#define SCENE_CUH

#include "common.cuh"
#include "triangle.cuh"
#include "light.cuh"
#include "bvh.cuh"
#include <iostream>
#include <chrono>

using namespace std::chrono;

/* Declarations */

enum SceneType {
    BUNNY, 
    ERATO, 
    DRAGON, 
    AURELIUS
};

struct Scene {
    Vector3 viewport;
    Vector3 cameraPosition, cameraRotation;
    Color backgroundColor;
    Triangle *triangles;
    size_t trianglesSize;
    double ambientLight;
    Point *pointLights;
    size_t pointLightsSize;
    Directional *directionalLights;
    size_t directionalLightsSize;
    BVH bvh;

    void initScene(SceneType sceneType);
    void initSceneBunny();
    void initSceneErato();
    void initSceneDragon();
    void initSceneAurelius();

    vector<Triangle> parseOBJ(string OBJPath);
    __device__ 
    Vector3 toViewport(int x, int y, int canvasWidth, int canvasHeight);
    __device__ 
    Color traceRay(
        Vector3 origin, Vector3 ray, double tMin, double tMax, int recursionDepth
    );
    __device__ 
    double computeTotalLighting(
        Vector3 point, Vector3 normal, Vector3 inverse, int specularity
    );
    __device__ 
    double computeLighting(
        double intensity, Vector3 light, double tMax, 
        Vector3 point, Vector3 normal, Vector3 inverse, int specularity
    );
    __device__ 
    Vector3 reflectRay(Vector3 ray, Vector3 normal);
};

/* Definitions */

void Scene::initScene(SceneType sceneType) {
    string OBJPath;

    switch (sceneType) {
        case BUNNY:    { OBJPath = "scenes/bunny.obj";    break; }
        case ERATO:    { OBJPath = "scenes/erato.obj";    break; }
        case DRAGON:   { OBJPath = "scenes/dragon.obj";   break; }
        case AURELIUS: { OBJPath = "scenes/aurelius.obj"; break; }
    }

    vector<Triangle> faces = parseOBJ(OBJPath);
    trianglesSize = faces.size();
    cudaMallocManaged(&triangles, trianglesSize * sizeof(Triangle));

    for (int i = 0; i < trianglesSize; i++) {
        triangles[i] = faces[i];
    }

    auto bmStart = high_resolution_clock::now();

    bvh = BVH(triangles, trianglesSize);

    auto bmStop = high_resolution_clock::now();
    auto bmDuration = duration_cast<milliseconds>(bmStop - bmStart);

    std::cout << "[ BVH ] " 
              << static_cast<double>(bmDuration.count()) / 1000 
              << " seconds.\n";

    directionalLightsSize = 1;
    cudaMallocManaged(
        &directionalLights, 
        directionalLightsSize * sizeof(Directional)
    );

    directionalLights[0] = { 0.5, { 0, 1, -1 } };

    cudaPrefetch(triangles, trianglesSize * sizeof(Triangle));
    cudaPrefetch(directionalLights, directionalLightsSize * sizeof(Directional));

    switch (sceneType) { 
        case BUNNY:    { initSceneBunny();    break; }
        case ERATO:    { initSceneErato();    break; }
        case DRAGON:   { initSceneDragon();   break; }
        case AURELIUS: { initSceneAurelius(); break; }
    }
}

/* 144056 faces */
void Scene::initSceneBunny() {
    viewport = {1, 1, 1};
    cameraPosition = {0.1425, 2, -5.94};
    cameraRotation = {0, 0.1, 0};
    backgroundColor = {255, 255, 255};
    ambientLight = 0.2;
    pointLights = NULL;
    pointLightsSize = 0;

    for (size_t i = 0; i < trianglesSize - 10; i++) {
        triangles[i].reflectivity = 0.2;
    }

    for (size_t i = trianglesSize - 10; i < trianglesSize - 4; i++) {
        triangles[i].reflectivity = 0.4;
    }

    for (size_t i = trianglesSize - 4; i < trianglesSize - 2; i++) {
        triangles[i].color = {0, 255, 0};
        triangles[i].reflectivity = 0.4;
    }

    for (size_t i = trianglesSize - 2; i < trianglesSize; i++) {
        triangles[i].color = {255, 0, 0};
        triangles[i].reflectivity = 0.4;
    }
}

/* 412508 faces */
void Scene::initSceneErato() {
    viewport = {1, 1, 1};
    cameraPosition = {-0.8, 28.7, -86.1};
    cameraRotation = {0, 0.1, 0};
    backgroundColor = {255, 255, 255};
    ambientLight = 0.2;
    pointLights = NULL;
    pointLightsSize = 0;

    for (size_t i = 0; i < trianglesSize - 10; i++) {
        triangles[i].reflectivity = 0.2;
    }

    for (size_t i = trianglesSize - 10; i < trianglesSize - 4; i++) {
        triangles[i].reflectivity = 0.4;
    }

    for (size_t i = trianglesSize - 4; i < trianglesSize - 2; i++) {
        triangles[i].color = {0, 255, 0};
        triangles[i].reflectivity = 0.4;
    }

    for (size_t i = trianglesSize - 2; i < trianglesSize; i++) {
        triangles[i].color = {255, 0, 0};
        triangles[i].reflectivity = 0.4;
    }
}

/* 871316 faces */
void Scene::initSceneDragon() {
    viewport = {1, 1, 1};
    cameraPosition = {-0.0425, 0.7, -2.975};
    cameraRotation = {0, 0.1, 0};
    backgroundColor = {255, 255, 255};
    ambientLight = 0.2;
    pointLights = NULL;
    pointLightsSize = 0;

    for (size_t i = 0; i < trianglesSize - 10; i++) {
        triangles[i].reflectivity = 0.2;
    }

    for (size_t i = trianglesSize - 10; i < trianglesSize - 4; i++) {
        triangles[i].reflectivity = 0.4;
    }

    for (size_t i = trianglesSize - 4; i < trianglesSize - 2; i++) {
        triangles[i].color = {0, 255, 0};
        triangles[i].reflectivity = 0.4;
    }

    for (size_t i = trianglesSize - 2; i < trianglesSize; i++) {
        triangles[i].color = {255, 0, 0};
        triangles[i].reflectivity = 0.4;
    }
}

/* 1704768 faces */
void Scene::initSceneAurelius() {
    viewport = {1, 1, 1};
    cameraPosition = {-0.025, 3.05, -20.9};
    cameraRotation = {0, 0.1, 0};
    backgroundColor = {255, 255, 255};
    ambientLight = 0.2;
    pointLights = NULL;
    pointLightsSize = 0;

    for (size_t i = 0; i < trianglesSize - 10; i++) {
        triangles[i].reflectivity = 0.2;
    }

    for (size_t i = trianglesSize - 10; i < trianglesSize - 4; i++) {
        triangles[i].reflectivity = 0.4;
    }

    for (size_t i = trianglesSize - 4; i < trianglesSize - 2; i++) {
        triangles[i].color = {0, 255, 0};
        triangles[i].reflectivity = 0.4;
    }

    for (size_t i = trianglesSize - 2; i < trianglesSize; i++) {
        triangles[i].color = {255, 0, 0};
        triangles[i].reflectivity = 0.4;
    }
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

__device__ 
Vector3 Scene::toViewport(int x, int y, int canvasWidth, int canvasHeight) {
    return {
        x * (this->viewport.x / canvasWidth), 
        y * (this->viewport.y / canvasHeight), 
        this->viewport.z
    };
}

__device__ 
Color Scene::traceRay(
    Vector3 origin, Vector3 ray, double tMin, double tMax, int recursionDepth
) {
    double closestT;
    int closestTriangleIndex;

    if (!bvh.intersectRay(&closestT, &closestTriangleIndex, origin, ray, tMin, tMax, 0)) {
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

__device__ 
double Scene::computeTotalLighting(
    Vector3 point, Vector3 normal, Vector3 inverse, int specularity
) {
    double totalIntensity = ambientLight;

    for (size_t i = 0; i < pointLightsSize; i++) {
        Point pointLight = pointLights[i];
        totalIntensity += computeLighting(
            pointLight.intensity, pointLight.position - point, 1, 
            point, normal, inverse, specularity
        );
    }

    for (size_t i = 0; i < directionalLightsSize; i++) {
        Directional directionalLight = directionalLights[i];
        totalIntensity += computeLighting(
            directionalLight.intensity, directionalLight.direction, INFINITY, 
            point, normal, inverse, specularity
        );
    }

    return totalIntensity;
}

__device__ 
double Scene::computeLighting(
    double intensity, Vector3 light, double tMax, 
    Vector3 point, Vector3 normal, Vector3 inverse, int specularity
) {
    double totalIntensity = 0.0;
    double closestT;
    int closestTriangleIndex;

    /* Shadow check */
    if (bvh.intersectRay(&closestT, &closestTriangleIndex, point, light, 0.001, tMax, 0)) {
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

__device__ 
Vector3 Scene::reflectRay(Vector3 ray, Vector3 normal) {
    return 2 * normal * normal.dot(ray) - ray;
}

#endif // SCENE_CUH
