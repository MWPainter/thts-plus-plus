#pragma once
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "mo/pareto_front.h"

#include <algorithm>
#include <set>
#include <unordered_set>


namespace thts::test {
    using namespace std;
    using namespace thts;

    /**
     * Helper to check that s1 is a subset of s2
    */
    template <typename T>
    bool set_subset(unordered_set<TaggedPoint<T>> s1, unordered_set<TaggedPoint<T>> s2) {
        for (TaggedPoint<T> p1 : s1) {
            auto it = find(s2.begin(), s2.end(), p1);
            if (it != s2.end()) {
                if (!it->equals(p1)) {
                    return false;
                }
                if (it->tag != p1.tag) {
                    return false;
                }
            }
        }
        return true;
    };

    /**
     * Helper to compare sets of unordered sets
    */
    template <typename T>
    bool set_equals(unordered_set<TaggedPoint<T>> s1, unordered_set<TaggedPoint<T>> s2) {
        return set_subset(s1,s2) && set_subset(s2,s1);
    };

    /**
     * ParetoFront subclass to add testing checks
     * If TaggedPoint breaks, then these funcitons will break to, don't think can get around this
     * 
     * To use pf_points either need to have the "using ParetoFront<T>::pf_points" line, or use this->pf_points:
     * https://stackoverflow.com/questions/62127901/simple-way-to-reference-member-variables-of-base-class-templates
    */
    template <typename T>
    struct TestableParetoFront : public ParetoFront<T> {
        using ParetoFront<T>::pf_points;

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
            vector<TaggedPoint<T>> pf_points_vec;
            pf_points_vec.insert(pf_points_vec.begin(), pf_points.begin(), pf_points.end());
            for (unsigned int i=0; i<pf_points_vec.size(); i++) {
                for (unsigned int j=i+1; j<pf_points_vec.size(); j++) {
                    if (pf_points_vec[i].equals(pf_points_vec[j])) {
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
            for (const TaggedPoint<T>& point : pf_points) {
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
        std::unordered_set<TaggedPoint<T>> public_prune(
            const std::unordered_set<TaggedPoint<T>>& ref_points, 
            const std::unordered_set<TaggedPoint<T>>& points) const 
        {
            return ParetoFront<T>::prune(ref_points, points);
        }

        /**
         * Public version of 'prune' for testing
        */
        std::unordered_set<TaggedPoint<T>> public_prune(const std::unordered_set<TaggedPoint<T>>& points) const {
            return ParetoFront<T>::prune(points);
        }
    };
}