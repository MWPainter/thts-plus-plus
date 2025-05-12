#pragma once
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "mo/data_structures/pareto_front.h"

#include <algorithm>
#include <set>
#include <unordered_set>


namespace thts::test {
    using namespace std;
    using namespace thts;

    /**
     * Helper for eigen vector equals
     */
    bool vec_eq(const Eigen::ArrayXd& u, const Eigen::ArrayXd& v) {
        return (u==v).all();
    }

    /**
     * Helper to check that s1 is a subset of s2
    */
    bool set_subset(unordered_set<Eigen::ArrayXd> s1, unordered_set<Eigen::ArrayXd> s2) {
        for (const Eigen::ArrayXd& p1 : s1) {
            bool found = false;
            for (const Eigen::ArrayXd& p2 : s2) {
                if (vec_eq(p1,p2)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    };

    /**
     * Helper to compare sets of unordered sets
    */
    bool set_equals(unordered_set<Eigen::ArrayXd> s1, unordered_set<Eigen::ArrayXd> s2) {
        return set_subset(s1,s2) && set_subset(s2,s1);
    };

    /**
     * ParetoFront subclass to add testing checks
     * 
     * To use pf_points either need to have the "using ParetoFront::pf_points" line, or use this->pf_points:
     * https://stackoverflow.com/questions/62127901/simple-way-to-reference-member-variables-of-base-class-templates
    */
    struct TestableParetoFront : public ParetoFront {
        using ParetoFront::pf_points;

        /**
         * Constructor, empty
        */
        TestableParetoFront() : ParetoFront() {}; 

        /**
         * Constructor, set of Tagged points
         * With an option to say if we know that the set of points is already a pareto front
        */
        TestableParetoFront(const std::unordered_set<Eigen::ArrayXd>& init_points, bool already_pareto_front=false) :
            ParetoFront(init_points, already_pareto_front) {};

        /**
         * Copy constructor
        */
        TestableParetoFront(const ParetoFront& pf) : 
            ParetoFront(pf) {};
        TestableParetoFront(const TestableParetoFront& pf) : 
            ParetoFront(pf) {};

        /**
         * Move constructor
        */
        TestableParetoFront(const ParetoFront&& pf) :
            ParetoFront(pf) {};
        TestableParetoFront(const TestableParetoFront&& pf) :
            ParetoFront(pf) {};

        /**
         * Checks pareto front doesn't contain any duplicate points
         * 
         * Realised at a later date that this is pointless. 'pf_points' is a hashset, so it will never add two points 
         * that are identical to each other...
         * 
         * But can't hurt to run extra stuff when testing if it doesn't take long
         * 
         * So keep just in case we ever change backend to a vector instead of a set
        */
        bool contains_duplicate_points() {
            vector<Eigen::ArrayXd> pf_points_vec;
            pf_points_vec.insert(pf_points_vec.begin(), pf_points.begin(), pf_points.end());
            for (unsigned int i=0; i<pf_points_vec.size(); i++) {
                for (unsigned int j=i+1; j<pf_points_vec.size(); j++) {
                    if (vec_eq(pf_points_vec[i],pf_points_vec[j])) {
                        return true;
                    }
                }
            }
            return false;
        };


        /**
         * Checks for Pareto Fronts
        */
        bool check_fits_expected(unordered_set<Eigen::ArrayXd>& points) {
            if (this->size() != points.size()) {
                return false; // not correct number of points in pf
            }
            if (contains_duplicate_points()) {
                return false; // pf shouldn't contain duplicate points
            }
            for (const Eigen::ArrayXd& point : pf_points) {
                // find 'point' in 'points'
                auto it = points.begin();
                for ( ; it != points.end(); it++) {
                    if (vec_eq(point,*it)) {
                        break;
                    }
                }
                if (it == points.end()) {
                    return false; // pf contains a point not in the expected pf
                }
            }
            return true; // passed all the checks
        };

        // /**
        //  * Public version of 'add_points' for testing
        // */
        // void public_add_points(const std::unordered_set<TaggedPoint>& points_to_add) {
        //     return add_points(points_to_add);
        // };

        /**
         * Public version of 'prune' for testing
        */
        std::unordered_set<Eigen::ArrayXd> public_prune(
            const std::unordered_set<Eigen::ArrayXd>& ref_points, 
            const std::unordered_set<Eigen::ArrayXd>& points) const 
        {
            return ParetoFront::prune(ref_points, points);
        }

        /**
         * Public version of 'prune' for testing
        */
        std::unordered_set<Eigen::ArrayXd> public_prune(const std::unordered_set<Eigen::ArrayXd>& points) const {
            return ParetoFront::prune(points);
        }
    };
}