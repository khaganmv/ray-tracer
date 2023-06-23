#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "common.cuh"

/* Declarations */

struct Triangle {
    Vector3 v0, v1, v2;
    Color color;
    Vector3 normal;
    int specularity;
    double reflectivity;
    Vector3 centroid;

    __host__ __device__ 
    Triangle(
        Vector3 _v0, Vector3 _v1, Vector3 _v2, 
        Color _color, int _specularity, double _reflectivity
    ) : v0(_v0), v1(_v1), v2(_v2), 
        color(_color), specularity(_specularity), reflectivity(_reflectivity) {
        Vector3 e01 = v1 - v0;
        Vector3 e02 = v2 - v0;
        normal = e01.cross(e02).normalize();
        centroid = (v0 + v1 + v2) / 3;
    }

    __device__ 
    bool intersectRay(double *t, Vector3 origin, Vector3 ray);
};

/* Definitions */

__device__ 
bool Triangle::intersectRay(double *t, Vector3 origin, Vector3 ray) {
    *t = 0.0;
    double normalDotRay = normal.dot(ray);

    if (fabs(normalDotRay) < EPSILON) {
        return false;
    }

    double d = -normal.dot(v0);
    *t = -(normal.dot(origin) + d) / normalDotRay;
    Vector3 point = origin + *t * ray;

    Vector3 e01 = v1 - v0;
    Vector3 e12 = v2 - v1;
    Vector3 e20 = v0 - v2;
    Vector3 e0p = point - v0;
    Vector3 e1p = point - v1;
    Vector3 e2p = point - v2;

    return normal.dot(e01.cross(e0p)) > EPSILON 
        && normal.dot(e12.cross(e1p)) > EPSILON 
        && normal.dot(e20.cross(e2p)) > EPSILON;
}

#endif // TRIANGLE_CUH
