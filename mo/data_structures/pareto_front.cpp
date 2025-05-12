#include "mo/data_structures/pareto_front.h"

#include "helper_templates.h"
#include "mo/mo_helper.h"

#include <iostream>
#include <stdexcept>

using namespace std;

/**
 * ParetoFront implementation
 */
namespace thts {
    /**
     * Constructor, empty
    */
    ParetoFront::ParetoFront() :
        pf_points()
    {
    };
    
    /**
     * Constructor, add points immediately, optionally specify that already pareto front
    */
    ParetoFront::ParetoFront(const unordered_set<Eigen::ArrayXd>& init_points, bool already_pareto_front) :
        pf_points(already_pareto_front ? init_points : prune(init_points))
    {
    };

    /**
     * Constructor that initialises it with a single tagged point
    */
    ParetoFront::ParetoFront(const Eigen::ArrayXd& heuristic_val) :
        pf_points()
    {
        pf_points.insert(heuristic_val);
    };

    /**
     * Copy constructor
    */
    ParetoFront::ParetoFront(const ParetoFront& pf) :
        pf_points(pf.pf_points) 
    {
    };

    /**
     * Move constructor
    */
    ParetoFront::ParetoFront(const ParetoFront&& pf) :
        pf_points(std::move(pf.pf_points)) 
    {
    };

    // /**
    //  * Adds points to 'pf_points'
    //  * 
    //  * 1. remove any points in 'points_to_add' that are dominated by other points in 'points_to_add'
    //  * 2. remove any points in 'pf_points' that are dominated by 'points_to_add'
    //  * 3. remove any points in 'points_to_add' that are dominated by points in 'pf_points'
    //  * 4. Assign new pf_points
    //  * 
    //  * N.B. this function is used from constructors, so need to handle a frequent case where pf_points.size() == 0
    // */
    // template <typename T>
    // void ParetoFront<T>::add_points(const unordered_set<TaggedPoint<T>>& points_to_add) 
    // {
    //     if (points_to_add.size() == 0) {
    //         return;
    //     }

    //     unordered_set<TaggedPoint<T>> pruned_points_to_add = prune(points_to_add);
    //     unordered_set<TaggedPoint<T>> pruned_points_to_keep;
    //     if (pf_points.size() > 0) {
    //         unordered_set<TaggedPoint<T>> pruned_points_to_keep = prune(pruned_points_to_add, pf_points);
    //         pruned_points_to_add = prune(pruned_points_to_keep, pruned_points_to_add);
    //     }
        
    //     unordered_set<TaggedPoint<T>> pruned_points = pruned_points_to_add;
    //     pruned_points.reserve(pruned_points_to_add.size() + pruned_points_to_keep);
    //     for (TaggedPoint<T>& point : pruned_points_to_keep) {
    //         pruned_points.insert(point);
    //     }
        
    //     pf_points = pruned_points;
    // };

    /**
     * If this point u (weakly) dominates another point v
     * If any index has in u lower value than v, then u doesnt dominate it
     * If all indices are greater than or equal, we dominate the other point if we are not equal
    */
    bool ParetoFront::weakly_pareto_dominates(const Eigen::ArrayXd& u, const Eigen::ArrayXd& v) {
        if (u.size() != v.size()) {
            throw runtime_error("Trying to use 'weakly_pareto_dominates' with vectors with different dims.");
        }
        return (u >= v).all();
    }

    /**
     * Returns the set of points from 'points' that are not weakly dominated by any points in 'ref_points'
    */
    unordered_set<Eigen::ArrayXd> ParetoFront::prune(
        const unordered_set<Eigen::ArrayXd>& ref_points, const unordered_set<Eigen::ArrayXd>& points)
    {
        if (ref_points.size() == 0 || points.size() == 0) {
            return unordered_set<Eigen::ArrayXd>(points);
        }

        unordered_set<Eigen::ArrayXd> new_set;
        new_set.reserve(points.size());

        for (const Eigen::ArrayXd& p_point : points) {
            bool is_dominated = false;
            for (const Eigen::ArrayXd& r_point : ref_points) {
                if (weakly_pareto_dominates(r_point, p_point)) {
                    is_dominated = true;
                    break;
                }
            }
            if (!is_dominated) {
                new_set.insert(p_point);
            }
        }
        
        return new_set;
    };

    /**
     * Returns the Pareto front from the set of 'points'
     * 
     * Uses an iterator to remove points while iterating over the set of points:
     * https://stackoverflow.com/questions/2874441/deleting-elements-from-stdset-while-iterating
     * 
     * This lets us correctly keep one vector if there are duplicate vectors in the set
     * 
     * auto type = set<TaggedPoint<T>>::iterator
     * Compiler needs it to be declared "typename set<TaggedPoint<T>>::iterator it=pruned_points.begin()" for some 
     * reason: https://stackoverflow.com/questions/610245/where-and-why-do-i-have-to-put-the-template-and-typename-keywords
     * 
     * For each point in the set, we search for if there is another unique point in the set that weakly dominates it, 
     * and if so remove it. Note that a point weakly dominates itself, so take care to avoid that case.
    */
    unordered_set<Eigen::ArrayXd> ParetoFront::prune(const unordered_set<Eigen::ArrayXd>& points) {
        unordered_set<Eigen::ArrayXd> pruned_points(points);
        
        for (auto it = pruned_points.begin(); it != pruned_points.end(); ) {
            bool is_dominated = false; 
            for (auto jt = pruned_points.begin(); jt != pruned_points.end(); jt++) {
                if (it == jt) {
                    continue;
                }
                if (weakly_pareto_dominates(*jt,*it)) {
                    is_dominated = true;
                    break;
                }
            }
            if (is_dominated) {
                it = pruned_points.erase(it);
            } else {
                it++;
            }
        }

        return pruned_points;
    }

    /**
     * Get the size of pf
    */
    size_t ParetoFront::size() const {
        return pf_points.size();
    }

    /**
     * Scale a pareto front
    */
    ParetoFront ParetoFront::scale(double scale) const
    {
        unordered_set<Eigen::ArrayXd> scaled_pf_points;
        scaled_pf_points.reserve(size());
        for (const Eigen::ArrayXd& point : pf_points) {
            scaled_pf_points.insert(point*scale);
        }
        return ParetoFront(scaled_pf_points, true);
    };

    /**
     * Union of two pareto fronts ('union' is a keyword in c++, so called this combine)
     * 
     * If have pfs U and V, then prune({u | u in U or u in V})
    */
    ParetoFront ParetoFront::combine(const ParetoFront& other) const {
        if (other.size() > size()) {
            return other.combine(*this);
        }
        if (size() == 0) {
            return ParetoFront(other);
        } else if (other.size() == 0) {
            return ParetoFront(*this);
        }
        
        // as already pareto fronts, only need to check if points dominated by the other pareto front
        // care here if this and other contain the same vector, as using weak Pareto domination
        unordered_set<Eigen::ArrayXd> pruned_points_one = prune(other.pf_points, pf_points);
        unordered_set<Eigen::ArrayXd> pruned_points_two = prune(pruned_points_one, other.pf_points);
        
        pruned_points_one.reserve(pruned_points_one.size() + pruned_points_two.size());
        for (const Eigen::ArrayXd& point : pruned_points_two) {
            pruned_points_one.insert(point);
        }

        return ParetoFront(pruned_points_one, true);
    };

    /**
     * Add two pareto fronts 
     * Just sums all combinations
    */
    ParetoFront ParetoFront::add(const ParetoFront& other) const 
    {
        if (pf_points.size() == 0) {
            return ParetoFront(other);
        } else if (other.pf_points.size() == 0) {
            return ParetoFront(*this);
        }

        unordered_set<Eigen::ArrayXd> summed_points;
        for (const Eigen::ArrayXd& point : pf_points) {
            for (const Eigen::ArrayXd& other_point : other.pf_points) {
                summed_points.insert(point + other_point);
            }
        }

        // constructor will prune points
        return ParetoFront(summed_points);
    };

    /**
     * Add vector to pareto front
    */
    ParetoFront ParetoFront::add(const Eigen::ArrayXd& v) const 
    {
        unordered_set<Eigen::ArrayXd> summed_points;
        for (const Eigen::ArrayXd& point : pf_points) {
            summed_points.insert(point + v);
        }
        return ParetoFront(summed_points, true);
    };

    /**
     * Return points to iterate over externally
     */
    const unordered_set<Eigen::ArrayXd>& ParetoFront::get_points() const
    {
        return pf_points;
    }
}

/**
 * Forward declare operator overloads and output stream function sepcialisations for ParetoFront
 * Output stream for debugging
*/
namespace std {
    using namespace thts;

    /**
     * Scale by vector
    */
    ParetoFront operator*(const ParetoFront& pf, double s) {
        return pf.scale(s);
    }
    
    ParetoFront operator*(double s, const ParetoFront& pf) {
        return pf.scale(s);
    }

    /**
     * Union of two pareto fronts
    */
    
    ParetoFront operator|(const ParetoFront& pf1, const ParetoFront& pf2) {
        return pf1.combine(pf2);
    }

    /**
     * Sum of pareto fronts
    */
    
    ParetoFront operator+(const ParetoFront& pf1, const ParetoFront& pf2) {
        return pf1.add(pf2);
    }

    /**
     * Add vector to pareto front
    */
    
    ParetoFront operator+(const ParetoFront& pf, const Eigen::ArrayXd& v) {
        return pf.add(v);
    }

    
    ParetoFront operator+(const Eigen::ArrayXd& v, const ParetoFront& pf) {
        return pf.add(v);
    }

    /**
     * Output stream
    */
    
    ostream& operator<<(ostream& os, const ParetoFront& pf) {
        os << "ParetoFront = {" << endl;
        for (const Eigen::ArrayXd& point : pf.get_points()) {
            os << point << endl;
        }
        os << "}";
        return os;
    }
}