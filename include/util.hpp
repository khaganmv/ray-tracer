#ifndef UTIL_HPP
#define UTIL_HPP

#include <cmath>

/* Declarations */

double toRadians(double degrees);

/* Definitions */

double toRadians(double degrees) {
    return (degrees / 180.0) * M_PI;
}

#endif // UTIL_HPP
