// Vector.h
#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <cmath>
#include <iostream>

class Vector {
public:
    static std::vector<double> subtractVector(const std::vector<double>& v1, const std::vector<double>& v2) {
        std::vector<double> result(v1.size());

        for (size_t i = 0; i < v1.size(); ++i) {
            result[i] = v1[i] - v2[i];
        }

        return result;
    }

    static std::vector<double> addVector(const std::vector<double>& v1, const std::vector<double>& v2) {
        std::vector<double> result(v1.size());

        for (size_t i = 0; i < v1.size(); ++i) {
            result[i] = v1[i] + v2[i];
        }

        return result;
    }

    static double norm(const std::vector<double>& v) {
        double l1 = 0.0;
        for (auto& val : v) {
            l1 += std::abs(val);
        }
        return l1;
    }

    static std::vector<double> multiply(double scale, const std::vector<double>& v) {
        std::vector<double> result(v.size());

        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i] * scale;
        }

        return result;
    }

    static void copy(std::vector<double>& target, const std::vector<double>& source) {
        target = source;  // In C++, you can directly assign vectors to copy them
    }
};

#endif // VECTOR_H
