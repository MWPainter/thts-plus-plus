#include "multi_objective/mo_helper.h"

#include <stdexcept>

namespace std {

    /**
     * Hash for Eigen::ArrayXd
    */
    template <>
    size_t hash<Eigen::ArrayXd>::operator()(const Eigen::ArrayXd& v) const {
        size_t cur_hash = 0;
        cur_hash = helper::hash_combine(cur_hash, v.size());
        for (int i=0; i<v.size(); i++) {
            cur_hash = helper::hash_combine(cur_hash, v(i));
        }
        return cur_hash; 
    }

    /**
     * Equals for Eigen::ArrayXd
    */
    template <>
    bool equal_to<Eigen::ArrayXd>::operator()(const Eigen::ArrayXd& u, const Eigen::ArrayXd& v) {
        if (u.size() != v.size()) {
            throw runtime_error("Cant compare eigen arrays of different sizes");
        }
        return (u == v).all();
    }
}