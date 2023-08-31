#pragma once

#include <cassert>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>


/**
 * A Point in a Pareto Front / Convex Hull
 * 
 * In practise, we want points to be 'tagged'. For example, it will be useful to tag points with the actions that were 
 * used to obtain that value.
 * 
 * Member variables:
 *      value: A vector value (i.e. the point)
 *      tag: A tag associated with this point
*/
template <typenmae T>
struct TaggedPoint {
    Eigen::VectorXd value;
    T tag;
    
    /**
     * Constructor
    */
    TaggedPoint(Eigen::VectorXd& value, T& tag) :
        value(value), tag(tag) 
    {   
    };

    /**
     * Copy constructor
    */
    TaggedPoint(TaggedPoint& other) :
        value(other.value), tag(other.tag)
    {
    };

    /**
     * If this point dominates another point
     * If any index has a lower value than 'other' we don't dominate it
     * If all indices are greater than or equal, we dominate the other point if we are not equal
    */
    bool dominates(TaggedPoint& other) {
        assert((value.size() == other.value.size()))
        for (int i=0; i<value.size(); i++) {
            if (value(i) < other.value(i)) {
                return false;
            }
        }
        return !equals(other);
    }

    /**
     * Equality
    */
    friend bool equals(TaggedPoint& other) {
        assert((value.size() == other.value.size()));
        for (int i=0; i<value.size(); i++) {
            if (value(i) != other.value(i)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Equality operator
    */
    friend bool operator==(TaggedPoint& a, TaggedPoint& b) {
        return a.equals(b);
        assert((a.value.size() == b.value.size()));
        for (int i=0; i<a.value.size(); i++) {
            if (a.value(i) != b.value(i)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Lexographical less than ordering of points.
     * If get to end of loop, then a.value == b.value, so return false.
    */
    friend bool operator<(TaggedPoint& a, TaggedPoint& b) {
        assert((a.value.size() == b.value.size()));
        for (int i=0; i<a.value.size(); i++) {
            if (a.value(i) < b.value(i)) {
                return true;
            } else if (a.value(i) > b.value(i)) {
                return false;
            }
        }
        return false;
    };
};

/**
 * Logic surrounding an implementation of a Pareto Front
 * 
 * Member variables:
 *      points: 
 *          The set of TaggedPoints in the Pareto Front
 *      ordered_points: 
 *          A vector of TaggedPoints that are lexographically ordered. If nullptr then it needs recomputing.
 *          
*/
template <typename T>
class ParetoFront {
    protected:
        std::unordered_set<TaggedPoint> points;

    public:
        /**
         * Constructor, empty
        */
        ParetoFront() :
            points()
        {
        };

        /**
         * Constructor, adding points immediately
        */
        ParetoFront(std::vector<std::pair<Eigen::VectorXd,T>>& init_points) :
            points()
        {  
            std::unordered_set<TaggedPoint> init_points_set;
            for (std::pair<Eigen::VectorXd,T> pr : init_points) {
                init_points_set.emplace(pr.first, pr.second);
            }
            add_points(init_points_set); 
        };

        /**
         * Constructor, adding points immediately, with one tag
        */
        ParetoFront(std::vector<Eigen::VectorXd>& init_points, T tag) :
            points()
        {  
            std::unordered_set<TaggedPoint> init_points_set;
            for (Eigen::VectorXd value : init_points) {
                init_points_set.emplace(value, tag);
            }
            add_points(init_points_set); 
        };

        /**
         * Copy constructor
        */
        ParetoFront(ParetoFront& pf) :
            points(pf.points) 
        {
        };

    protected:
        /**
         * Private constrctor from a set of points
         * Assumes points are appropriately pruned, hence why it's not a public method
        */
        ParetoFront(std::unordered_set<TaggedPoint>& init_points) :
            points(init_points)
        {
        };

        /**
         * Adds points to 'points'
         * 
         * 1 convert input to set
         * 2 remove any points in 'points_to_add' that are dominated by points also in 'points_to_add'
         * 3 if 'points' empty, set it to the pruned points and return
         * 4 remove any points in 'points_to_add' that are dominated by points in 'points'
         * 5 add new points to 'points'
        */
        void add_points(std::unordered_set<TaggedPoint>& points_to_add) 
        {
            points_to_add = prune(points_to_add, points_to_add);
            if (points.size() == 0) {
                points = points_to_add;
                return;
            }
            points_to_add = prune(points, points_to_add);
            for (TaggedPoint& point : points_to_add) {
                points.insert(point);
            }
        };

        /**
         * Returns the set of points from 'points' that are not dominated by any points in 'ref_points'
        */
        std::unordered_set<TaggedPoint> prune(
            std::unordered_set<TaggedPoint>& ref_points, std::unordered_set<TaggedPoint>& points) 
        {
            std::unordered_set<TaggedPoint> new_set;
            new_set.reserve(points.size());

            for (TaggedPoint& p_point : points) {
                bool is_dominated = false;
                for (TaggedPoint& rp_point : ref_points) {
                    if (rp_point.dominates(p_point)) {
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

    public:
        /**
         * Scale a pareto front
        */
        ParetoFront scale(double scale) 
        {
            ParetoFront pf(*this);
            for (TaggedPoint& point : points) {
                point.value *= scale;
            }
            return pf;
        };

        /**
         * Union of two pareto fronts ('union' is a keyword in c++, so called this combine)
         * 
         * If have pfs U and V, then prune({u | u in U or u in V})
        */
        ParetoFront combine(ParetoFront& other) {
            if (points.size() == 0) {
                return ParetoFront(other);
            } else if (other.points.size() == 0) {
                return ParetoFront(*this);
            }
            
            // as already pareto fronts, only need to check if points dominated by the other pareto front
            std::unordered_set<TaggedPoint> pruned_points_one = prune(points, other.points);
            std::unordered_set<TaggedPoint> pruned_points_two = prune(other.points, points);
            
            pruned_points_one.reserve(pruned_points_one.size() + pruned_points_two.size());
            for (TaggedPoint& point : pruned_points_two) {
                pruned_points_one.insert(point);
            }
        };

        /**
         * Add two pareto fronts 
         * 
         * TODO: copy description
         * 
         * If one of the pareto fronts 
        */
        ParetoFront add(ParetoFront& other) 
        {
            if (points.size() == 0) {
                return ParetoFront(other);
            } else if (other.points.size() == 0) {
                return ParetoFront(*this);
            }

            // TODO: add all of the points together
            // TODO: prune
        };

        // TODO: add vector, no pruning required

};

// TODO: make convex hull bit
// TODO: use linear program lib for convex hull pruning: https://www.alglib.net/download.php
// https://github.com/ori-goals/rapport-algorithms/blob/mike-thts/rapport_algorithms/thts/utils/pareto_front.py