#ifndef VECTOR3_HPP
#define VECTOR3_HPP

#include "util.hpp"
#include <cmath>
#include <ostream>

/* Declarations */

struct Vector3 {
    double x, y, z;

    Vector3() = default;
    Vector3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
    
    double dot(Vector3 that);
    Vector3 cross(Vector3 that);

    Vector3 rotateX(double degrees);
    Vector3 rotateY(double degrees);
    Vector3 rotateZ(double degrees);

    double magnitude();
    Vector3 normalize();

    Vector3 operator-();
};

Vector3 operator+(Vector3 lhs, Vector3 rhs);
Vector3 operator-(Vector3 lhs, Vector3 rhs);
Vector3 operator*(double lhs, Vector3 rhs);
Vector3 operator*(Vector3 lhs, double rhs);
Vector3 operator/(Vector3 lhs, double rhs);
std::ostream& operator<<(std::ostream& lhs, Vector3 rhs);

/* Definitions */

double Vector3::dot(Vector3 that) {
    return this->x * that.x + this->y * that.y + this->z * that.z;
}

Vector3 Vector3::cross(Vector3 that) {
    return {
        this->y * that.z - this->z * that.y, 
        this->z * that.x - this->x * that.z, 
        this->x * that.y - this->y * that.x
    };
}

Vector3 Vector3::rotateX(double degrees) {
    double radians = toRadians(degrees);
    
    return {
        this->x, 
        cos(radians) * this->y - sin(radians) * this->z, 
        sin(radians) * this->y + cos(radians) * this->z
    };
}

Vector3 Vector3::rotateY(double degrees) {
    double radians = toRadians(degrees);

    return {
        cos(radians) * this->x + sin(radians) * this->z, 
        this->y, 
        -sin(radians) * this->x + cos(radians) * this->z
    };
}

Vector3 Vector3::rotateZ(double degrees) {
    double radians = toRadians(degrees);

    return {
        cos(radians) * this->x - sin(radians) * this->y, 
        sin(radians) * this->x + cos(radians) * this->y, 
        this->z
    };
}

double Vector3::magnitude() {
    return sqrt(this->dot(*this));
}

Vector3 Vector3::normalize() {
    return *this / this->magnitude();
}

Vector3 Vector3::operator-() {
    return *this * -1;
}

Vector3 operator+(Vector3 lhs, Vector3 rhs) {
    return {
        lhs.x + rhs.x, 
        lhs.y + rhs.y, 
        lhs.z + rhs.z
    };
}

Vector3 operator-(Vector3 lhs, Vector3 rhs) {
    return {
        lhs.x - rhs.x, 
        lhs.y - rhs.y, 
        lhs.z - rhs.z
    };
}

Vector3 operator*(double lhs, Vector3 rhs) {
    return {
        lhs * rhs.x, 
        lhs * rhs.y, 
        lhs * rhs.z
    };
}

Vector3 operator*(Vector3 lhs, double rhs) {
    return {
        lhs.x * rhs, 
        lhs.y * rhs, 
        lhs.z * rhs
    };
}

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

#endif // VECTOR3_HPP
