#pragma once
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "test/mo/test_pareto_front.h"

#include "mo/convex_hull.h"

#include <algorithm>
#include <set>
#include <unordered_set>


namespace thts::test {
    using namespace std;
    using namespace thts;

    /**
     * Helpers defined in test_pareto_front.h
     * 
     * set_subset
     * set_equals
    */

    /**
     * ConvexHull subclass to add testing checks
     * If TaggedPoint breaks, then these funcitons will break to, don't think can get around this
     * 
     * To use ch_points either need to have the "using ConvexHull<T>::ch_points" line, or use this->ch_points:
     * https://stackoverflow.com/questions/62127901/simple-way-to-reference-member-variables-of-base-class-templates
    */
    template <typename T>
    struct TestableConvexHull : public ConvexHull<T> {
        using ConvexHull<T>::ch_points;

        /**
         * Constructor, empty
        */
        TestableConvexHull() : ConvexHull<T>() {}; 

        /**
         * Constructor, adding points immediately
        */
        TestableConvexHull(const std::vector<std::pair<Eigen::ArrayXd,T>>& init_points) :
            ConvexHull<T>(init_points) {};

        /**
         * Constructor, adding points immediately, with one tag
        */
        TestableConvexHull(const std::vector<Eigen::ArrayXd>& init_points, const T& tag) :
            ConvexHull<T>(init_points, tag) {};

        /**
         * Constructor, set of Tagged points
         * With an option to say if we know that the set of points is already a pareto front
        */
        TestableConvexHull(const std::unordered_set<TaggedPoint<T>>& init_points, bool already_pareto_front=false) :
            ConvexHull<T>(init_points, already_pareto_front) {};

        /**
         * Copy constructor
        */
        TestableConvexHull(const ConvexHull<T>& pf) : 
            ConvexHull<T>(pf) {};
        TestableConvexHull(const TestableConvexHull<T>& pf) : 
            ConvexHull<T>(pf) {};

        /**
         * Move constructor
        */
        TestableConvexHull(const ConvexHull<T>&& pf) :
            ConvexHull<T>(pf) {};
        TestableConvexHull(const TestableConvexHull<T>&& pf) :
            ConvexHull<T>(pf) {};

        /**
         * Checks pareto front doesn't contain any duplicate points
         * 
         * Realised at a later date that this is pointless. 'ch_points' is a hashset, so it will never add two points 
         * that are identical to each other...
         * 
         * But can't hurt to run extra stuff when testing if it doesn't take long
         * 
         * So keep just in case we ever change backend to a vector instead of a set
        */
        bool contains_duplicate_points() {
            vector<TaggedPoint<T>> ch_points_vec;
            ch_points_vec.insert(ch_points_vec.begin(), ch_points.begin(), ch_points.end());
            for (unsigned int i=0; i<ch_points_vec.size(); i++) {
                for (unsigned int j=i+1; j<ch_points_vec.size(); j++) {
                    if (ch_points_vec[i].equals(ch_points_vec[j])) {
                        return true;
                    }
                }
            }
            return false;
        };


        /**
         * Checks for Pareto Fronts
        */
        bool check_fits_expected_multitag(unordered_set<TaggedPoint<unordered_set<T>>>& points) {
            if (this->size() != points.size()) {
                return false; // not correct number of points in pf
            }
            if (contains_duplicate_points()) {
                return false; // pf shouldn't contain duplicate points
            }
            for (const TaggedPoint<T>& point : ch_points) {
                // find 'point' in 'points'
                auto it = points.begin();
                for ( ; it != points.end(); it++) {
                    if ((point.point == it->point).all()) {
                        break;
                    }
                }
                if (it == points.end()) {
                    return false; // pf contains a point not in the expected pf
                }
                if (!it->tag.contains(point.tag)) {
                    return false; // point in pf doesn't contain any of the possible correct tags
                }
            }
            return true; // passed all the checks
        };

        /**
         * Checks for Pareto Fronts, where only have one tag (override function)
        */
        bool check_fits_expected(unordered_set<TaggedPoint<T>>& points) {
            unordered_set<TaggedPoint<unordered_set<T>>> expanded_points;
            for (const TaggedPoint<T>& point : points) {
                unordered_set<T> extended_tags = {point.tag};
                TaggedPoint<unordered_set<T>> extended_point(point.point,extended_tags);
                expanded_points.insert(extended_point);
            }
            return check_fits_expected_multitag(expanded_points);
        };

        // /**
        //  * Public version of 'add_points' for testing
        // */
        // void public_add_points(const std::unordered_set<TaggedPoint<T>>& points_to_add) {
        //     return add_points(points_to_add);
        // };

        /**
         * Public version of 'prune' for testing
        */
        // std::unordered_set<TaggedPoint<T>> public_prune(
        //     const std::unordered_set<TaggedPoint<T>>& ref_points, 
        //     const std::unordered_set<TaggedPoint<T>>& points) const 
        // {
        //     return ConvexHull<T>::prune(ref_points, points);
        // }

        /**
         * Public version of 'prune' for testing
        */
        std::unordered_set<TaggedPoint<T>> public_prune(const std::unordered_set<TaggedPoint<T>>& points) const {
            return ConvexHull<T>::prune(points);
        }
    };
}