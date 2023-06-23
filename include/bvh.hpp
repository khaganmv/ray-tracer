#ifndef BVH_HPP
#define BVH_HPP

#include "common.hpp"
#include "triangle.hpp"

/* Declarations */

struct BVHNode {
    Vector3 AABBMin, AABBMax;
    size_t triangleFirst, triangleCount;
};

struct BVH {
    vector<Triangle> triangles;
    vector<size_t> indices;
    BVHNode *nodes;
    size_t nodesUsed;

    BVH() = default;
    BVH(vector<Triangle> _triangles);

    void updateNodeBounds(size_t index);
    void subdivide(size_t index);

    tuple<bool, double, int> intersectRay(
        Vector3 origin, Vector3 ray, 
        double tMin, double tMax, 
        size_t index
    );
    bool intersectRayAABB(
        Vector3 origin, Vector3 ray, 
        Vector3 bMin, Vector3 bMax, 
        double tMax
    );
};

/* Definitions */

BVH::BVH(vector<Triangle> _triangles) {
    triangles = _triangles;

    for (size_t i = 0; i < _triangles.size(); i++) {
        indices.push_back(i);
    }

    nodes = new BVHNode[2 * _triangles.size() - 1];
    nodes[0].triangleFirst = 0;
    nodes[0].triangleCount = _triangles.size();
    nodesUsed = 1;

    updateNodeBounds(0);
    subdivide(0);
}

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

void BVH::subdivide(size_t index) {
    BVHNode &node = nodes[index];

    if (node.triangleCount <= 2) {
        return;
    }

    Vector3 extent = node.AABBMax - node.AABBMin;
    int axis = 0;

    if (extent.y > extent.x) {
        axis = 1;
    }

    if (extent.z > extent[axis]) {
        axis = 2;
    }

    double split = node.AABBMin[axis] + extent[axis] / 2;

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

tuple<bool, double, int> BVH::intersectRay(
    Vector3 origin, Vector3 ray, 
    double tMin, double tMax, 
    size_t index
) {
    bool intersectsAny = false;
    double closestT = INFINITY;
    int closestTriangleIndex = -1;
    BVHNode &node = nodes[index];

    if (!intersectRayAABB(origin, ray, node.AABBMin, node.AABBMax, closestT)) {
        return { false, INFINITY, -1 };
    }

    if (node.triangleCount > 0) {
        for (size_t i = 0; i < node.triangleCount; i++) {
            size_t triangleIndex = indices[node.triangleFirst + i];
            tuple<bool, double> intersection = triangles[triangleIndex].intersectRay(origin, ray);
            bool intersects = std::get<0>(intersection);
            double _t = std::get<1>(intersection);

            if (intersects) {
                if (_t > tMin && _t < tMax && _t < closestT) {
                    intersectsAny = true;
                    closestT = _t;
                    closestTriangleIndex = static_cast<int>(triangleIndex);
                }
            }
        }

        return { intersectsAny, closestT, closestTriangleIndex };
    }

    tuple<bool, double, int> first = intersectRay(
        origin, ray, tMin, tMax, node.triangleFirst
    );
    tuple<bool, double, int> second = intersectRay(
        origin, ray, tMin, tMax, node.triangleFirst + 1
    );
    
    intersectsAny = std::get<0>(first) || std::get<0>(second);
    closestT = std::min(std::get<1>(first), std::get<1>(second));
    closestTriangleIndex = (std::get<1>(first) < std::get<1>(second)) 
                            ? std::get<2>(first) 
                            : std::get<2>(second);

    return { intersectsAny, closestT, closestTriangleIndex };
}

bool BVH::intersectRayAABB(
    Vector3 origin, Vector3 ray, 
    Vector3 bMin, Vector3 bMax, 
    double tMax
) {
    double tx1 = (bMin.x - origin.x) / ray.x;
    double tx2 = (bMax.x - origin.x) / ray.x;
    double tMin = std::min(tx1, tx2);
    double tMax = std::max(tx1, tx2);
    
    double ty1 = (bMin.y - origin.y) / ray.y;
    double ty2 = (bMax.y - origin.y) / ray.y;
    tMin = std::max(tMin, std::min(ty1, ty2));
    tMax = std::min(tMax, std::max(ty1, ty2));

    double tz1 = (bMin.z - origin.z) / ray.z;
    double tz2 = (bMax.z - origin.z) / ray.z;
    tMin = std::max(tMin, std::min(tz1, tz2));
    tMax = std::min(tMax, std::max(tz1, tz2));

    return tMin < tMax && tMax > 0 && tMax >= tMin;
}

#endif // BVH_HPP
