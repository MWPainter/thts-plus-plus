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
     * Helper to check that s1 is a subset of s2
    */
    bool set_subset(unordered_set<Vec> s1, unordered_set<Vec> s2);

    /**
     * Helper to compare sets of unordered sets
    */
    bool set_equals(unordered_set<Vec> s1, unordered_set<Vec> s2);

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
        TestableParetoFront(const std::unordered_set<Vec>& init_points, bool already_pareto_front=false) :
            ParetoFront(init_points, already_pareto_front) {};
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
            vector<Vec> pf_points_vec;
            pf_points_vec.insert(pf_points_vec.begin(), pf_points.begin(), pf_points.end());
            for (unsigned int i=0; i<pf_points_vec.size(); i++) {
                for (unsigned int j=i+1; j<pf_points_vec.size(); j++) {
                    if (pf_points_vec[i] == pf_points_vec[j]) {
                        return true;
                    }
                }
            }
            return false;
        };


        /**
         * Checks for Pareto Fronts
        */
        bool check_fits_expected(unordered_set<Vec>& points) {
            if (this->size() != points.size()) {
                return false; // not correct number of points in pf
            }
            if (contains_duplicate_points()) {
                return false; // pf shouldn't contain duplicate points
            }
            for (const Vec& point : pf_points) {
                // find 'point' in 'points'
                auto it = points.begin();
                for ( ; it != points.end(); it++) {
                    if (*it == point) {
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
        std::unordered_set<Vec> public_prune(
            const std::unordered_set<Vec>& ref_points,
            const std::unordered_set<Vec>& points) const
        {
            return ParetoFront::prune(ref_points, points);
        }

        /**
         * Public version of 'prune' for testing
        */
        std::unordered_set<Vec> public_prune(const std::unordered_set<Vec>& points) const {
            return ParetoFront::prune(points);
        }
    };
}