#ifndef SCENE_CUH
#define SCENE_CUH

#include "common.cuh"
#include "triangle.cuh"
#include "light.cuh"

/* Declarations */

enum SceneType {
    TEAPOT, 
    SUZANNE, 
    BUNNY, 
    TYRANNOSAURUS 
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

    void initScene(SceneType sceneType);
    void initSceneTeapot();
    void initSceneSuzanne();
    void initSceneBunny();
    void initSceneTyrannosaurus();

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
    __device__ 
    bool closestIntersection(
        double *closestT, int *closestTriangleIndex, 
        Vector3 origin, Vector3 ray, double tMin, double tMax
    );
};

/* Definitions */

void Scene::initScene(SceneType sceneType) {
    string OBJPath;

    switch (sceneType) {
        case TEAPOT:        { OBJPath = "scenes/teapot.obj";        break; }
        case SUZANNE:       { OBJPath = "scenes/suzanne.obj";       break; }
        case BUNNY:         { OBJPath = "scenes/bunny.obj";         break; }
        case TYRANNOSAURUS: { OBJPath = "scenes/tyrannosaurus.obj"; break; }
    }

    vector<Triangle> faces = parseOBJ(OBJPath);
    trianglesSize = faces.size();
    cudaMallocManaged(&triangles, trianglesSize * sizeof(Triangle));

    for (int i = 0; i < trianglesSize; i++) {
        triangles[i] = faces[i];
    }

    directionalLightsSize = 1;
    cudaMallocManaged(
        &directionalLights, 
        directionalLightsSize * sizeof(Directional)
    );

    switch (sceneType) {
        case TEAPOT:        { directionalLights[0] = { 0.5, {-1, 0, -1} }; break; }
        case SUZANNE:       { directionalLights[0] = { 0.5, { 1, 0,  3} }; break; }
        case BUNNY:         { directionalLights[0] = { 0.5, { 1, 0,  5} }; break; }
        case TYRANNOSAURUS: { directionalLights[0] = { 0.5, { 1, 1,  1} }; break; }
    }

    cudaPrefetch(triangles, trianglesSize * sizeof(Triangle));
    cudaPrefetch(directionalLights, directionalLightsSize * sizeof(Directional));

    switch (sceneType) { 
        case TEAPOT:        { initSceneTeapot();        break; }
        case SUZANNE:       { initSceneSuzanne();       break; }
        case BUNNY:         { initSceneBunny();         break; }
        case TYRANNOSAURUS: { initSceneTyrannosaurus(); break; }
    }
}

/* 6320 faces */
void Scene::initSceneTeapot() {
    viewport = {1, 1, 1};
    cameraPosition = {3, 2, -8};
    cameraRotation = {0, -20, 0};
    backgroundColor = {0, 0, 0};
    ambientLight = 0.2;
    pointLights = NULL;
    pointLightsSize = 0;
}

/* 15488 faces */
void Scene::initSceneSuzanne() {
    viewport = {1, 1, 1};
    cameraPosition = {-1, 0, 4};
    cameraRotation = {0, 165, 0};
    backgroundColor = {0, 0, 0};
    ambientLight = 0.2;
    pointLights = NULL;
    pointLightsSize = 0;
}

/* 69630 faces */
void Scene::initSceneBunny() {
    viewport = {1, 1, 1};
    cameraPosition = {-0.4, 1.25, 6};
    cameraRotation = {0, 180, 0};
    backgroundColor = {0, 0, 0};
    ambientLight = 0.2;
    pointLights = NULL;
    pointLightsSize = 0;
}

/* 200000 faces */
void Scene::initSceneTyrannosaurus() {
    viewport = {1, 1, 1};
    cameraPosition = {3, 0, -4};
    cameraRotation = {0, -45, 0};
    backgroundColor = {0, 0, 0};
    ambientLight = 0.2;
    pointLights = NULL;
    pointLightsSize = 0;
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
                    1, 
                    -1
                }
            );
        }
    }

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

    if (!closestIntersection(&closestT, &closestTriangleIndex, origin, ray, tMin, tMax)) {
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
    if (closestIntersection(&closestT, &closestTriangleIndex, point, light, 0.001, tMax)) {
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

__device__ 
bool Scene::closestIntersection(
    double *closestT, int *closestTriangleIndex, 
    Vector3 origin, Vector3 ray, double tMin, double tMax
) {
    bool intersectsAny = false;
    *closestT = INFINITY;
    *closestTriangleIndex = -1;

    for (size_t i = 0; i < trianglesSize; i++) {
        double t;

        if (triangles[i].intersectRay(&t, origin, ray)) {
            if (t > tMin && t < tMax && t < *closestT) {
                intersectsAny = true;
                *closestT = t;
                *closestTriangleIndex = static_cast<int>(i);
            }
        }
    }

    return intersectsAny;
}

#endif // SCENE_CUH
