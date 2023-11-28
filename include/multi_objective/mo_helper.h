#pragma once

#include <Eigen/Dense>

/**
 * Hash and equal_to for Eigen::ArrayXd
*/
namespace std {
    using namespace thts;

    /**
     * Hash
    */
    template <>
    struct hash<Eigen::ArrayXd> {
        size_t operator()(const Eigen::ArrayXd&) const;
    };

    /**
     * Equals
    */
    template <>
    struct equal_to<Eigen::ArrayXd> {
        bool operator()(const Eigen::ArrayXd&, const Eigen::ArrayXd&);
    };
}