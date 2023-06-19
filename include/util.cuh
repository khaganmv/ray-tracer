#ifndef UTIL_CUH
#define UTIL_CUH

#include <cmath>

/* Declarations */

__device__ 
double toRadians(double degrees);
void cudaPrefetch(void *ptr, size_t n);

/* Definitions */

__device__ 
double toRadians(double degrees) {
    return (degrees / 180.0) * M_PI;
}

void cudaPrefetch(void *ptr, size_t n) {
    int device = -1;

    cudaGetDevice(&device);
    cudaMemPrefetchAsync(ptr, n, device, NULL);
}

#endif // UTIL_CUH
