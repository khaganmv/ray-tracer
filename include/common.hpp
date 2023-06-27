#ifndef COMMON_HPP
#define COMMON_HPP

#include "vector3.hpp"
#include "color.hpp"
#include <tuple>
#include <array>
#include <vector>
#include <string>
#include <fstream>

#define EPSILON std::numeric_limits<double>::epsilon() * 1048576.0

#define USE_BVH 0

using std::tuple;
using std::array;
using std::vector;
using std::string;
using std::fstream;

#endif // COMMON_HPP
