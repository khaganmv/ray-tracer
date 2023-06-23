#ifndef BVH_CUH
#define BVH_CUH

#include "common.cuh"
#include "triangle.cuh"

/* Declarations */

struct AABB {
    Vector3 bMin, bMax;

    __host__ 
    AABB() : bMin(Vector3(INFINITY, INFINITY, INFINITY)), 
             bMax(Vector3(-INFINITY, -INFINITY, -INFINITY)) {}

    __host__ 
    void grow(Vector3 p);
    __host__ 
    double area();
};

struct BVHNode {
    Vector3 AABBMin, AABBMax;
    size_t triangleFirst, triangleCount;
};

struct BVH {
    Triangle *triangles;
    size_t trianglesSize;
    size_t *indices;
    size_t indicesSize;
    BVHNode *nodes;
    size_t nodesUsed;

    __host__ 
    BVH() = default;
    __host__ 
    BVH(Triangle *_triangles, size_t _trianglesSize);

    __host__ 
    void updateNodeBounds(size_t index);
    __host__ 
    void subdivide(size_t index);

    __device__ 
    bool intersectRay(
        double *closestT, int *closestTriangleIndex, 
        Vector3 origin, Vector3 ray, 
        double tMin, double tMax, 
        size_t index
    );
    __device__ 
    bool intersectRayAABB(
        Vector3 origin, Vector3 ray, 
        Vector3 bMin, Vector3 bMax, 
        double tMax
    );
    double evaluateSAH(size_t index, int axis, double pos);
};

/* Definitions */

__host__ 
void AABB::grow(Vector3 p) {
    bMin = bMin.min(p);
    bMax = bMax.max(p);
}

__host__ 
double AABB::area() {
    Vector3 extent = bMax - bMin;
    
    // *2 test
    return extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
}

__host__ 
BVH::BVH(Triangle *_triangles, size_t _trianglesSize) {
    triangles = _triangles;
    trianglesSize = indicesSize = _trianglesSize;

    cudaMallocManaged(&indices, indicesSize * sizeof(size_t));

    for (size_t i = 0; i < indicesSize; i++) {
        indices[i] = i;
    }

    cudaMallocManaged(&nodes, (2 * trianglesSize + 1) * sizeof(BVHNode));

    nodes[0].triangleFirst = 0;
    nodes[0].triangleCount = trianglesSize;
    nodesUsed = 1;

    updateNodeBounds(0);
    subdivide(0);
}

__host__ 
void BVH::updateNodeBounds(size_t index) {
    BVHNode &node = nodes[index];
    node.AABBMin = Vector3(INFINITY, INFINITY, INFINITY);
    node.AABBMax = Vector3(-INFINITY, -INFINITY, -INFINITY);

    for (size_t first = node.triangleFirst, i = 0; i < node.triangleCount; i++) {
        size_t j = indices[first + i];
        Triangle& leaf = triangles[j];
        node.AABBMin = node.AABBMin.min(leaf.v0);
        node.AABBMin = node.AABBMin.min(leaf.v1);
        node.AABBMin = node.AABBMin.min(leaf.v2);
        node.AABBMax = node.AABBMax.max(leaf.v0);
        node.AABBMax = node.AABBMax.max(leaf.v1);
        node.AABBMax = node.AABBMax.max(leaf.v2);
    }
}

__host__ 
void BVH::subdivide(size_t index) {
    BVHNode &node = nodes[index];

    Vector3 extent = node.AABBMax - node.AABBMin;
    double parentArea = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
    double parentCost = node.triangleCount * parentArea;

    int bestAxis = -1;
    double bestPos = 0;
    double bestCost = INFINITY;

    for (int axis = 0; axis < 3; axis++) {
        for (size_t i = 0; i < node.triangleCount; i++) {
            Triangle &triangle = triangles[indices[node.triangleFirst + i]];
            double candidatePos = triangle.centroid[axis];
            double cost = evaluateSAH(index, axis, candidatePos);

            if (cost < bestCost) {
                bestPos = candidatePos;
                bestAxis = axis;
                bestCost = cost;
            }
        }
    }

    if (bestCost >= parentCost) {
        return;
    }

    int axis = bestAxis;
    double split = bestPos;

    int i = static_cast<int>(node.triangleFirst);
    int j = static_cast<int>(i + node.triangleCount - 1);

    while (i <= j) {
        if (triangles[indices[i]].centroid[axis] < split) {
            i++;
        } else {
            size_t temp = indices[j];
            indices[j] = indices[i];
            indices[i] = temp;
            j--;
        }
    }

    int trianglesLeft = static_cast<int>(i - node.triangleFirst);

    if (trianglesLeft == 0 || trianglesLeft == static_cast<int>(node.triangleCount)) {
        return;
    }

    int leftChild = static_cast<int>(nodesUsed++);
    int rightChild = static_cast<int>(nodesUsed++);

    nodes[leftChild].triangleFirst = node.triangleFirst;
    nodes[leftChild].triangleCount = trianglesLeft;
    nodes[rightChild].triangleFirst = i;
    nodes[rightChild].triangleCount = node.triangleCount - trianglesLeft;
    node.triangleFirst = leftChild;
    node.triangleCount = 0;

    updateNodeBounds(leftChild);
    updateNodeBounds(rightChild);
    subdivide(leftChild);
    subdivide(rightChild);
}

__device__
bool BVH::intersectRay(
    double *closestT, int *closestTriangleIndex, 
    Vector3 origin, Vector3 ray, 
    double tMin, double tMax, 
    size_t index
) {
    bool intersectsAny = false;
    *closestT = INFINITY;
    *closestTriangleIndex = -1;
    BVHNode &node = nodes[index];

    if (!intersectRayAABB(origin, ray, node.AABBMin, node.AABBMax, *closestT)) {
        return false;
    }

    if (node.triangleCount > 0) {
        for (size_t i = 0; i < node.triangleCount; i++) {
            size_t triangleIndex = indices[node.triangleFirst + i];
            double _t;
            bool intersects = triangles[triangleIndex].intersectRay(&_t, origin, ray);

            if (intersects) {
                if (_t > tMin && _t < tMax && _t < *closestT) {
                    intersectsAny = true;
                    *closestT = _t;
                    *closestTriangleIndex = static_cast<int>(triangleIndex);
                }
            }
        }

        return intersectsAny;
    }

    double firstT;
    int firstClosestTriangleIndex;
    bool firstIntersects = intersectRay(
        &firstT, &firstClosestTriangleIndex, 
        origin, ray, 
        tMin, tMax, 
        node.triangleFirst
    );

    double secondT;
    int secondClosestTriangleIndex;
    bool secondIntersects = intersectRay(
        &secondT, &secondClosestTriangleIndex, 
        origin, ray, 
        tMin, tMax, 
        node.triangleFirst + 1
    );
    
    intersectsAny = firstIntersects || secondIntersects;
    *closestT = min(firstT, secondT);
    *closestTriangleIndex = (firstT < secondT) 
                            ? firstClosestTriangleIndex 
                            : secondClosestTriangleIndex;

    return intersectsAny;
}

__device__
bool BVH::intersectRayAABB(
    Vector3 origin, Vector3 ray, 
    Vector3 bMin, Vector3 bMax, 
    double tLim
) {
    double tx1 = (bMin.x - origin.x) / ray.x;
    double tx2 = (bMax.x - origin.x) / ray.x;
    double tMin = min(tx1, tx2);
    double tMax = max(tx1, tx2);
    
    double ty1 = (bMin.y - origin.y) / ray.y;
    double ty2 = (bMax.y - origin.y) / ray.y;
    tMin = max(tMin, min(ty1, ty2));
    tMax = min(tMax, max(ty1, ty2));

    double tz1 = (bMin.z - origin.z) / ray.z;
    double tz2 = (bMax.z - origin.z) / ray.z;
    tMin = max(tMin, min(tz1, tz2));
    tMax = min(tMax, max(tz1, tz2));

    return tMin < tLim && tMax > 0 && tMax >= tMin;
}

double BVH::evaluateSAH(size_t index, int axis, double pos) {
    BVHNode &node = nodes[index];
    AABB leftBox, rightBox;
    int leftCount = 0, rightCount = 0;

    for (size_t i = 0; i < node.triangleCount; i++) {
        Triangle &triangle = triangles[indices[node.triangleFirst + i]];

        if (triangle.centroid[axis] < pos) {
            leftCount++;
            leftBox.grow(triangle.v0);
            leftBox.grow(triangle.v1);
            leftBox.grow(triangle.v2);
        } else {
            rightCount++;
            rightBox.grow(triangle.v0);
            rightBox.grow(triangle.v1);
            rightBox.grow(triangle.v2);
        }
    }

    double cost = leftCount * leftBox.area() + rightCount * rightBox.area();

    return (cost > 0) ? cost : INFINITY;
}

#endif // BVH_CUH
