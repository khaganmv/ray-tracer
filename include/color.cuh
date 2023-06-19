#ifndef COLOR_CUH
#define COLOR_CUH

/* Declarations */

struct Color {
    int r, g, b;

    __host__ __device__ 
    Color() = default;
    __host__ __device__ 
    Color(int _r, int _g, int _b) : r(_r), g(_g), b(_b) {}

    __host__ __device__ 
    Color normalize();
};

__host__ __device__ 
Color operator+(Color lhs, Color rhs);
__host__ __device__ 
Color operator*(double lhs, Color rhs);
__host__ __device__ 
Color operator*(Color lhs, double rhs);

/* Definitions */

__host__ __device__ 
Color Color::normalize() {
    return {
        min(max(this->r, 0), 255), 
        min(max(this->g, 0), 255),
        min(max(this->b, 0), 255)
    };
}

__host__ __device__ 
Color operator+(Color lhs, Color rhs) {
    return {
        lhs.r + rhs.r, 
        lhs.g + rhs.g, 
        lhs.b + rhs.b
    };
}

__host__ __device__ 
Color operator*(double lhs, Color rhs) {
    return {
        static_cast<int>(lhs * rhs.r), 
        static_cast<int>(lhs * rhs.g), 
        static_cast<int>(lhs * rhs.b)
    };
}

__host__ __device__ 
Color operator*(Color lhs, double rhs) {
    return {
        static_cast<int>(lhs.r * rhs), 
        static_cast<int>(lhs.g * rhs), 
        static_cast<int>(lhs.b * rhs)
    };
}

#endif // COLOR_CUH
