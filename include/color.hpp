#ifndef COLOR_HPP
#define COLOR_HPP

#include <algorithm>

/* Declarations */

struct Color {
    int r, g, b;

    Color() = default;
    Color(int _r, int _g, int _b) : r(_r), g(_g), b(_b) {}

    Color normalize();
};

Color operator+(Color lhs, Color rhs);
Color operator*(double lhs, Color rhs);
Color operator*(Color lhs, double rhs);

/* Definitions */

Color Color::normalize() {
    return {
        std::min(std::max(this->r, 0), 255), 
        std::min(std::max(this->g, 0), 255),
        std::min(std::max(this->b, 0), 255)
    };
}

Color operator+(Color lhs, Color rhs) {
    return {
        lhs.r + rhs.r, 
        lhs.g + rhs.g, 
        lhs.b + rhs.b
    };
}

Color operator*(double lhs, Color rhs) {
    return {
        static_cast<int>(lhs * rhs.r), 
        static_cast<int>(lhs * rhs.g), 
        static_cast<int>(lhs * rhs.b)
    };
}

Color operator*(Color lhs, double rhs) {
    return {
        static_cast<int>(lhs.r * rhs), 
        static_cast<int>(lhs.g * rhs), 
        static_cast<int>(lhs.b * rhs)
    };
}

#endif // COLOR_HPP
