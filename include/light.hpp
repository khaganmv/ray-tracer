#ifndef LIGHT_HPP
#define LIGHT_HPP

#include "vector3.hpp"

/* Declarations */

struct Point {
    double intensity;
    Vector3 position;
};

struct Directional {
    double intensity;
    Vector3 direction;
};

#endif // LIGHT_HPP
