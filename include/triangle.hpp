#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "common.hpp"

/* Declarations */

struct Triangle {
    Vector3 v0, v1, v2;
    Color color;
    Vector3 normal;
    int specularity;

    Triangle(
        Vector3 _v0, Vector3 _v1, Vector3 _v2, Color _color, int _specularity
    ) : v0(_v0), v1(_v1), v2(_v2), color(_color), specularity(_specularity) {
        Vector3 e01 = this->v1 - this->v0;
        Vector3 e02 = this->v2 - this->v0;
        this->normal = e01.cross(e02);
    }

    tuple<bool, double> intersectRay(Vector3 origin, Vector3 ray);
};

/* Definitions */

tuple<bool, double> Triangle::intersectRay(Vector3 origin, Vector3 ray) {
    double normalDotRay = this->normal.dot(ray);

    if (fabs(normalDotRay) < EPSILON) {
        return { false, 0.0 };
    }

    double d = -this->normal.dot(this->v0);
    double t = -(this->normal.dot(origin) + d) / normalDotRay;
    Vector3 point = origin + t * ray;

    Vector3 e01 = this->v1 - this->v0;
    Vector3 e12 = this->v2 - this->v1;
    Vector3 e20 = this->v0 - this->v2;
    Vector3 e0p = point - this->v0;
    Vector3 e1p = point - this->v1;
    Vector3 e2p = point - this->v2;

    return {
        normal.dot(e01.cross(e0p)) > EPSILON && 
        normal.dot(e12.cross(e1p)) > EPSILON && 
        normal.dot(e20.cross(e2p)) > EPSILON, t
    };
}

#endif // TRIANGLE_HPP
