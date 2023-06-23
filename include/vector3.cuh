#ifndef VECTOR3_CUH
#define VECTOR3_CUH

#include "util.cuh"
#include <cmath>
#include <ostream>

/* Declarations */

struct Vector3 {
    double x, y, z;

    __host__ __device__ 
    Vector3() = default;
    __host__ __device__ 
    Vector3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
    
    __host__ __device__ 
    double dot(Vector3 that);
    __host__ __device__ 
    Vector3 cross(Vector3 that);

    __device__ 
    Vector3 rotateX(double degrees);
    __device__ 
    Vector3 rotateY(double degrees);
    __device__ 
    Vector3 rotateZ(double degrees);

    __host__ __device__ 
    double magnitude();
    __host__ __device__ 
    Vector3 normalize();

    __host__ __device__ 
    Vector3 min(Vector3 that);
    __host__ __device__ 
    Vector3 max(Vector3 that);

    __device__ 
    Vector3 operator-();
    __host__ __device__ 
    double operator[](size_t index);
};

__host__ __device__ 
Vector3 operator+(Vector3 lhs, Vector3 rhs);
__host__ __device__ 
Vector3 operator-(Vector3 lhs, Vector3 rhs);
__device__ 
Vector3 operator*(double lhs, Vector3 rhs);
__device__ 
Vector3 operator*(Vector3 lhs, double rhs);
__host__ __device__ 
Vector3 operator/(Vector3 lhs, double rhs);
std::ostream& operator<<(std::ostream& lhs, Vector3 rhs);

/* Definitions */

__host__ __device__ 
double Vector3::dot(Vector3 that) {
    return this->x * that.x + this->y * that.y + this->z * that.z;
}

__host__ __device__ 
Vector3 Vector3::cross(Vector3 that) {
    return {
        this->y * that.z - this->z * that.y, 
        this->z * that.x - this->x * that.z, 
        this->x * that.y - this->y * that.x
    };
}

__device__ 
Vector3 Vector3::rotateX(double degrees) {
    double radians = toRadians(degrees);
    
    return {
        this->x, 
        cos(radians) * this->y - sin(radians) * this->z, 
        sin(radians) * this->y + cos(radians) * this->z
    };
}

__device__ 
Vector3 Vector3::rotateY(double degrees) {
    double radians = toRadians(degrees);

    return {
        cos(radians) * this->x + sin(radians) * this->z, 
        this->y, 
        -sin(radians) * this->x + cos(radians) * this->z
    };
}

__device__ 
Vector3 Vector3::rotateZ(double degrees) {
    double radians = toRadians(degrees);

    return {
        cos(radians) * this->x - sin(radians) * this->y, 
        sin(radians) * this->x + cos(radians) * this->y, 
        this->z
    };
}

__host__ __device__ 
double Vector3::magnitude() {
    return sqrt(this->dot(*this));
}

__host__ __device__ 
Vector3 Vector3::normalize() {
    return *this / this->magnitude();
}

__host__ __device__ 
Vector3 Vector3::min(Vector3 that) {
    return {
        (x < that.x) ? x : that.x, 
        (y < that.y) ? y : that.y, 
        (z < that.z) ? z : that.z
    };
}

__host__ __device__ 
Vector3 Vector3::max(Vector3 that) {
    return {
        (x > that.x) ? x : that.x, 
        (y > that.y) ? y : that.y, 
        (z > that.z) ? z : that.z
    };
}

__device__ 
Vector3 Vector3::operator-() {
    return *this * -1;
}

__host__ __device__ 
double Vector3::operator[](size_t index) {
    if (index == 0) {
        return x;
    } else if (index == 1) {
        return y;
    } else {
        return z;
    }
}

__host__ __device__ 
Vector3 operator+(Vector3 lhs, Vector3 rhs) {
    return {
        lhs.x + rhs.x, 
        lhs.y + rhs.y, 
        lhs.z + rhs.z
    };
}

__host__ __device__ 
Vector3 operator-(Vector3 lhs, Vector3 rhs) {
    return {
        lhs.x - rhs.x, 
        lhs.y - rhs.y, 
        lhs.z - rhs.z
    };
}

__device__ 
Vector3 operator*(double lhs, Vector3 rhs) {
    return {
        lhs * rhs.x, 
        lhs * rhs.y, 
        lhs * rhs.z
    };
}

__device__ 
Vector3 operator*(Vector3 lhs, double rhs) {
    return {
        lhs.x * rhs, 
        lhs.y * rhs, 
        lhs.z * rhs
    };
}

__host__ __device__ 
Vector3 operator/(Vector3 lhs, double rhs) {
    return {
        lhs.x / rhs, 
        lhs.y / rhs, 
        lhs.z / rhs
    };
}

std::ostream& operator<<(std::ostream& lhs, Vector3 rhs) {
    return lhs << "x: " << rhs.x << " y: " << rhs.y << " z: " << rhs.z;
}

#endif // VECTOR3_CUH
