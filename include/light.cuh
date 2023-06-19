#ifndef LIGHT_CUH
#define LIGHT_CUH

#include "vector3.cuh"

/* Declarations */

struct Point {
    double intensity;
    Vector3 position;
};

struct Directional {
    double intensity;
    Vector3 direction;
};

#endif // LIGHT_CUH
