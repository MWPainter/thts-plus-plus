#pragma once
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "mo/data_structures/convex_hull.h"
#include "test/mo/test_pareto_front.h"

#include <algorithm>
#include <set>
#include <unordered_set>


namespace thts::test {
    using namespace std;
    using namespace thts;

    /**
     * Helpers (imported from pareto front tests)
     * 
     * set_subset
     * set_equals
    */

    /**
     * ConvexHull subclass to add testing checks
     * If Vec breaks, then these functions will break too, don't think can get around this
     * 
     * To use ch_points either need to have the "using ConvexHull::ch_points" line, or use this->ch_points:
     * https://stackoverflow.com/questions/62127901/simple-way-to-reference-member-variables-of-base-class-templates
    */
    struct TestableConvexHull : public ConvexHull {
        using ConvexHull::ch_points;

        /**
         * Constructor, empty
        */
        TestableConvexHull() : ConvexHull() {}; 

        /**
         * Constructor, set of Vec points
         * With an option to say if we know that the set of points is already a convex hull
        */
        TestableConvexHull(const std::unordered_set<Vec>& init_points, bool already_convex_hull=false) :
            ConvexHull(init_points, -1, 1e-9, already_convex_hull) {};

        /**
         * Copy constructor
        */
        TestableConvexHull(const ConvexHull& ch) : 
            ConvexHull(ch) {};
        TestableConvexHull(const TestableConvexHull& ch) : 
            ConvexHull(ch) {};

        /**
         * Move constructor (matching base class signature with const)
        */
        TestableConvexHull(const ConvexHull&& ch) :
            ConvexHull(std::move(ch)) {};
        TestableConvexHull(const TestableConvexHull&& ch) :
            ConvexHull(std::move(ch)) {};

        /**
         * Copy assignment operator
        */
        TestableConvexHull& operator=(const TestableConvexHull& ch) {
            ConvexHull::operator=(ch);
            return *this;
        }

        /**
         * Move assignment operator - accept const rvalue (non-const rvalue from std::move can bind to this)
        */
        TestableConvexHull& operator=(const TestableConvexHull&& ch) {
            // Bind to base class const reference, then pass to base move assignment
            // This converts const TestableConvexHull&& -> const ConvexHull& -> const ConvexHull&& via std::move
            const ConvexHull& base_lref = ch;
            ConvexHull::operator=(std::move(base_lref));
            return *this;
        }

        /**
         * Checks convex hull doesn't contain any duplicate points
         * 
         * Realised at a later date that this is pointless. 'ch_points' is a hashset, so it will never add two points 
         * that are identical to each other...
         * 
         * But can't hurt to run extra stuff when testing if it doesn't take long
         * 
         * So keep just in case we ever change backend to a vector instead of a set
        */
        bool contains_duplicate_points() {
            vector<Vec> ch_points_vec;
            ch_points_vec.insert(ch_points_vec.begin(), ch_points.begin(), ch_points.end());
            for (unsigned int i=0; i<ch_points_vec.size(); i++) {
                for (unsigned int j=i+1; j<ch_points_vec.size(); j++) {
                    if (ch_points_vec[i] == ch_points_vec[j]) {
                        return true;
                    }
                }
            }
            return false;
        };


        /**
         * Checks for Convex Hulls
        */
        bool check_fits_expected(unordered_set<Vec>& points) {
            if (this->size() != points.size()) {
                return false; // not correct number of points in ch
            }
            if (contains_duplicate_points()) {
                return false; // ch shouldn't contain duplicate points
            }
            for (const Vec& point : ch_points) {
                // find 'point' in 'points'
                auto it = points.begin();
                for ( ; it != points.end(); it++) {
                    if (point == *it) {
                        break;
                    }
                }
                if (it == points.end()) {
                    return false; // ch contains a point not in the expected ch
                }
            }
            return true; // passed all the checks
        };

        /**
         * Public version of 'prune' for testing
        */
        std::unordered_set<Vec> public_prune(const std::unordered_set<Vec>& points) const {
            return ConvexHull::prune(points);
        }

        /**
         * Public version of 'strongly_convex_dominated' for testing
        */
        static bool public_strongly_convex_dominated(
            const std::unordered_set<Vec>& ref_points, 
            const Vec& point) 
        {
            return ConvexHull::strongly_convex_dominated(ref_points, point);
        }
    };
}
