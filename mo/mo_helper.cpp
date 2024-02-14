#include "mo/mo_helper.h"

#include "helper_templates.h"

#include <stdexcept>


namespace std {

    /**
     * Hash for Eigen::ArrayXd
    */
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
    bool equal_to<Eigen::ArrayXd>::operator()(const Eigen::ArrayXd& u, const Eigen::ArrayXd& v) {
        if (u.size() != v.size()) {
            throw runtime_error("Cant compare eigen arrays of different sizes");
        }
        return (u == v).all();
    }
}

namespace thts::helper {
    double dist(const Eigen::ArrayXd& p1, const Eigen::ArrayXd& p2) {
        Eigen::ArrayXd delta = p1 - p2;
        return sqrt(delta.pow(2.0).sum());
    }
    double dot(const Eigen::ArrayXd& p1, const Eigen::ArrayXd& p2) {
        return (p1 * p2).sum();
    }

    
    ConstHeuristicFn::ConstHeuristicFn(Eigen::ArrayXd& const_val) : const_val(const_val)
    {
    };
    
    Eigen::ArrayXd ConstHeuristicFn::heuristic_fn(std::shared_ptr<const State> s, std::shared_ptr<ThtsEnv> env) 
    {
        return const_val;
    }
}