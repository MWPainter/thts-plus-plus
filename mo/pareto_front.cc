#include "mo/pareto_front.h"

#include "helper_templates.h"

#include <iostream>
#include <stdexcept>

using namespace std;

/**
 * TaggedPoint implementation
 */
namespace thts {
    
    /**
     * TaggedPoint Constructor
    */
    template <typename T>
    TaggedPoint<T>::TaggedPoint(const Eigen::ArrayXd& point, const T& tag) :
        point(point), tag(tag) 
    {   
    };

    /**
     * TaggedPoint Copy constructor
    */
    template <typename T>
    TaggedPoint<T>::TaggedPoint(const TaggedPoint<T>& other) :
        point(other.point), tag(other.tag)
    {
    };

    /**
     * TaggedPoint Move constructor
    */
    template <typename T>
    TaggedPoint<T>::TaggedPoint(const TaggedPoint<T>&& other) :
        point(move(other.point)), tag(move(other.tag))
    {
    };

    /**
     * TaggedPoint copy assignment
    */
    template <typename T>
    TaggedPoint<T>& TaggedPoint<T>::operator=(const TaggedPoint<T>& other) {
        point = other.point;
        tag = other.tag;
        return *this;
    }

    /**
     * TaggedPoint move assignment
    */
    template <typename T>
    TaggedPoint<T>& TaggedPoint<T>::operator=(const TaggedPoint<T>&& other) {
        point = move(other.point);
        tag = move(other.tag);
        return *this;
    }

    /**
     * If this point dominates another point
     * If any index has a lower value than 'other' we don't dominate it
     * If all indices are greater than or equal, we dominate the other point if we are not equal
    */
    template <typename T>
    bool TaggedPoint<T>::weakly_pareto_dominates(const TaggedPoint<T>& other) const {
        if (point.size() != other.point.size()) {
            throw runtime_error("Trying to use 'weakly_pareto_dominates' with vectors with different dims.");
        }
        return (point >= other.point).all();
    }

    /**
     * Equality
    */
    template <typename T>
    bool TaggedPoint<T>::equals(const TaggedPoint<T>& other) const {
        if (point.size() != other.point.size()) {
            throw runtime_error("Trying to compare vectors with different dims.");
        }
        return (point == other.point).all();
    }

    /**
     * Equality operator
    */
    template <typename T>
    bool TaggedPoint<T>::operator==(const TaggedPoint<T>& other) const {
        return equals(other);
    }

    /**
     * Hash value
    */
    template<typename T>
    size_t TaggedPoint<T>::hash() const {
        size_t cur_hash = 0;
        cur_hash = helper::hash_combine(cur_hash, point.size());
        for (int i=0; i<point.size(); i++) {
            cur_hash = helper::hash_combine(cur_hash, point[i]);
        }
        return cur_hash; 
    }
}

/**
 * ParetoFront implementation
 */
namespace thts {
    /**
     * Constructor, empty
    */
    template <typename T>
    ParetoFront<T>::ParetoFront() :
        pf_points()
    {
    };

    /**
     * Constructor, adding points immediately
    */
    template <typename T>
    ParetoFront<T>::ParetoFront(const vector<pair<Eigen::ArrayXd,T>>& init_points) :
        pf_points()
    {  
        unordered_set<TaggedPoint<T>> tagged_init_points;
        tagged_init_points.reserve(init_points.size());
        for (const pair<Eigen::ArrayXd,T>& pr : init_points) {
            // TaggedPoint<T> tp(pr.first,pr.second);
            // tagged_init_points.insert(tp);
            // tagged_init_points.insert(TaggedPoint<T>(pr.first,pr.second));
            tagged_init_points.emplace(pr.first,pr.second);
        }
        pf_points = prune(tagged_init_points);
    };

    /**
     * Constructor, adding points immediately, with one tag
    */
    template <typename T>
    ParetoFront<T>::ParetoFront(const vector<Eigen::ArrayXd>& init_points, const T& tag) :
        pf_points()
    {  
        unordered_set<TaggedPoint<T>> tagged_init_points;
        tagged_init_points.reserve(init_points.size());
        for (const Eigen::ArrayXd& point : init_points) {
            tagged_init_points.insert(TaggedPoint<T>(point,tag));
        }
        pf_points = prune(tagged_init_points);
    };
    
    /**
     * Constructor, set of Tagged points
     * Note that not assuming that 'init_points' is a Pareto Front, do use 'add_points' function to prune
    */
    template <typename T>
    ParetoFront<T>::ParetoFront(const unordered_set<TaggedPoint<T>>& init_points, bool already_pareto_front) :
        pf_points(already_pareto_front ? init_points : prune(init_points))
    {
    };

    /**
     * Constructor that initialises it with a single tagged point
    */
    template <typename T>
    ParetoFront<T>::ParetoFront(const Eigen::ArrayXd& heuristic_val, const T& tag) :
        pf_points()
    {
        pf_points.insert(TaggedPoint<T>(heuristic_val, tag));
    };

    /**
     * Copy constructor
    */
    template <typename T>
    ParetoFront<T>::ParetoFront(const ParetoFront<T>& pf) :
        pf_points(pf.pf_points) 
    {
    };

    /**
     * Move constructor
    */
    template <typename T>
    ParetoFront<T>::ParetoFront(const ParetoFront<T>&& pf) :
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
     * Returns the set of points from 'points' that are not weakly dominated by any points in 'ref_points'
    */
    template <typename T>
    unordered_set<TaggedPoint<T>> ParetoFront<T>::prune(
        const unordered_set<TaggedPoint<T>>& ref_points, const unordered_set<TaggedPoint<T>>& points) const
    {
        if (ref_points.size() == 0 || points.size() == 0) {
            return unordered_set<TaggedPoint<T>>(points);
        }

        unordered_set<TaggedPoint<T>> new_set;
        new_set.reserve(points.size());

        for (const TaggedPoint<T>& p_point : points) {
            bool is_dominated = false;
            for (const TaggedPoint<T>& rp_point : ref_points) {
                if (rp_point.weakly_pareto_dominates(p_point)) {
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
    template <typename T>
    unordered_set<TaggedPoint<T>> ParetoFront<T>::prune(const unordered_set<TaggedPoint<T>>& points) const {
        unordered_set<TaggedPoint<T>> pruned_points(points);
        
        for (auto it = pruned_points.begin(); it != pruned_points.end(); ) {
            bool is_dominated = false; 
            for (auto jt = pruned_points.begin(); jt != pruned_points.end(); jt++) {
                if (it == jt) {
                    continue;
                }
                if (jt->weakly_pareto_dominates(*it)) {
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
    template<typename T>
    size_t ParetoFront<T>::size() const {
        return pf_points.size();
    }

    /**
     * Set tag of all points in pf
    */
    template<typename T>
    void ParetoFront<T>::set_tags(const T& new_tag) {
        for (const TaggedPoint<T>& point : pf_points) {
            point.tag = new_tag;
        }
    }

    /**
     * Scale a pareto front
    */
    template <typename T>
    ParetoFront<T> ParetoFront<T>::scale(double scale) const
    {
        unordered_set<TaggedPoint<T>> scaled_pf_points;
        scaled_pf_points.reserve(size());
        for (const TaggedPoint<T>& point : pf_points) {
            scaled_pf_points.insert(TaggedPoint<T>(point.point*scale, point.tag));
        }
        return ParetoFront<T>(scaled_pf_points, true);
    };

    /**
     * Union of two pareto fronts ('union' is a keyword in c++, so called this combine)
     * 
     * If have pfs U and V, then prune({u | u in U or u in V})
    */
    template <typename T>
    ParetoFront<T> ParetoFront<T>::combine(const ParetoFront<T>& other) const {
        if (other.size() > size()) {
            return other.combine(*this);
        }
        if (size() == 0) {
            return ParetoFront<T>(other);
        } else if (other.size() == 0) {
            return ParetoFront<T>(*this);
        }
        
        // as already pareto fronts, only need to check if points dominated by the other pareto front
        // care here if this and other contain the same vector, as using weak Pareto domination
        unordered_set<TaggedPoint<T>> pruned_points_one = prune(other.pf_points, pf_points);
        unordered_set<TaggedPoint<T>> pruned_points_two = prune(pruned_points_one, other.pf_points);
        
        pruned_points_one.reserve(pruned_points_one.size() + pruned_points_two.size());
        for (const TaggedPoint<T>& point : pruned_points_two) {
            pruned_points_one.insert(point);
        }

        return ParetoFront<T>(pruned_points_one, true);
    };

    /**
     * Add two pareto fronts 
     * Just sums all combinations
    */
    template <typename T>
    ParetoFront<T> ParetoFront<T>::add(const ParetoFront<T>& other) const 
    {
        if (pf_points.size() == 0) {
            return ParetoFront<T>(other);
        } else if (other.pf_points.size() == 0) {
            return ParetoFront<T>(*this);
        }

        unordered_set<TaggedPoint<T>> summed_points;
        for (const TaggedPoint<T>& point : pf_points) {
            for (const TaggedPoint<T>& other_point : other.pf_points) {
                summed_points.insert(TaggedPoint<T>(point.point + other_point.point, point.tag));
            }
        }

        // constructor will prune points
        return ParetoFront<T>(summed_points);
    };

    /**
     * Add vector to pareto front
    */
    template <typename T>
    ParetoFront<T> ParetoFront<T>::add(const Eigen::ArrayXd& v) const 
    {
        unordered_set<TaggedPoint<T>> summed_points;
        for (const TaggedPoint<T>& point : pf_points) {
            summed_points.insert(TaggedPoint<T>(point.point + v, point.tag));
        }
        return ParetoFront<T>(summed_points, true);
    };
}



/**
 * std namespace function declarations for TaggedPoint
*/
namespace std {
    using namespace thts;

    /**
     * Hash
    */
    template <typename T>
    size_t hash<TaggedPoint<T>>::operator()(const TaggedPoint<T>& point) const {
        return point.hash();
    }

    /**
     * Equals
    */
    template <typename T>
    bool equal_to<TaggedPoint<T>>::operator()(const TaggedPoint<T>& u, const TaggedPoint<T>& v) const {
        return u.equals(v);
    }

    /**
     * Output stream
    */
    template <typename T>
    ostream& operator<<(ostream& os, const TaggedPoint<T>& point) {
        os << "(";
        for (int i=0; i<point.point.size(); i++) {
            os << point.point[i];
            if (i != point.point.size()-1) {
                os << ",";
            }
        }
        os << "|" << point.tag << ")";
        return os;
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
    template <typename T>
    ParetoFront<T> operator*(const ParetoFront<T>& pf, double s) {
        return pf.scale(s);
    }
    
    template <typename T>
    ParetoFront<T> operator*(double s, const ParetoFront<T>& pf) {
        return pf.scale(s);
    }

    /**
     * Union of two pareto fronts
    */
    template <typename T>
    ParetoFront<T> operator|(const ParetoFront<T>& pf1, const ParetoFront<T>& pf2) {
        return pf1.combine(pf2);
    }

    /**
     * Sum of pareto fronts
    */
    template <typename T>
    ParetoFront<T> operator+(const ParetoFront<T>& pf1, const ParetoFront<T>& pf2) {
        return pf1.add(pf2);
    }

    /**
     * Add vector to pareto front
    */
    template <typename T>
    ParetoFront<T> operator+(const ParetoFront<T>& pf, const Eigen::ArrayXd& v) {
        return pf.add(v);
    }

    template <typename T>
    ParetoFront<T> operator+(const Eigen::ArrayXd& v, const ParetoFront<T>& pf) {
        return pf.add(v);
    }

    /**
     * Output stream
    */
    template <typename T>
    ostream& operator<<(ostream& os, const ParetoFront<T>& pf) {
        os << "ParetoFront = {" << endl;
        for (TaggedPoint<T> point : pf.pf_points) {
            os << point << endl;
        }
        os << "}";
        return os;
    }
}