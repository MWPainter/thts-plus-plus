#pragma once

#include "mo/pareto_front.h"

#include <unordered_set>
#include <utility>

#include <Eigen/Dense>

namespace thts {
    /**
     * Convex Hull implementation
     * 
     * TODO: use description have in local branch, copy implementations from here
     * TODO: add comments for functions that have different imlpementations to pareto fronts
     * TODO: also add comments saying extra things that we could implement
     *      - use qhul convex hull (adding a reference point)
     *      - computing the hypervolume using convex hull (will actually need to do this for hypervolume indicator act selection)
     *          
    */
    template <typename T>
    class ConvexHull {
        protected:
            std::unordered_set<TaggedPoint<T>> ch_points;

        public:
            ConvexHull();
            ConvexHull(const std::vector<std::pair<Eigen::ArrayXd,T>>& init_points);
            ConvexHull(const std::vector<Eigen::ArrayXd>& init_points, const T& tag);
            ConvexHull(const std::unordered_set<TaggedPoint<T>>& init_points, bool already_pareto_front=false);
            ConvexHull(const Eigenn::ArrayXd& heuristic_val);
            ConvexHull(const ConvexHull<T>& pf);
            ConvexHull(const ConvexHull<T>&& pf);

        protected:
            /**
             * Returns if 'point' is dominated by any ppints in 'ref_points', which is checked using a linear program
             * This is a strong domination compared to the weak notion we used in ParetoFront
            */
            bool strongly_convex_dominated(const std::unordered_set<TaggedPoint<T>>& ref_points, TaggedPoint<T>& point);
            /**
             * Main functions that are overriden from Pareto Front
            */
            std::unordered_set<TaggedPoint<T>> prune(
                const std::unordered_set<TaggedPoint<T>>& ref_points, 
                const std::unordered_set<TaggedPoint<T>>& points) const;
            std::unordered_set<TaggedPoint<T>> prune(const std::unordered_set<TaggedPoint<T>>& points) const;

        public:
            std::size_t size() const;
            void set_tags(const T& new_tag);
            ConvexHull<T> scale(double scale) const;
            ConvexHull<T> combine(const ConvexHull<T>& other) const;
            ConvexHull<T> add(const ConvexHull<T>& other) const;
            ConvexHull<T> add(const Eigen::ArrayXd& v) const;

    };
}

/**
 * Forward declare operator overloads and output stream function sepcialisations for ConvexHull
 * Output stream for debugging
*/
namespace std {
    using namespace thts;

    /**
     * Scale by vector
    */
    template <typename T>
    ConvexHull<T> operator*(const ConvexHull<T>& pf, double s);
    
    template <typename T>
    ConvexHull<T> operator*(double s, const ConvexHull<T>& pf);

    /**
     * Union of two pareto fronts
    */
    template <typename T>
    ConvexHull<T> operator|(const ConvexHull<T>& pf1, const ConvexHull<T>& pf2);

    /**
     * Sum of pareto fronts
    */
    template <typename T>
    ConvexHull<T> operator+(const ConvexHull<T>& pf1, const ConvexHull<T>& pf2);

    /**
     * Add vector to pareto front
    */
    template <typename T>
    ConvexHull<T> operator+(const ConvexHull<T>& pf, const Eigen::ArrayXd& v);

    template <typename T>
    ConvexHull<T> operator+(const Eigen::ArrayXd& v, const ConvexHull<T>& pf);

    /**
     * Output stream
    */
    template <typename T>
    ostream& operator<<(ostream& os, const ConvexHull<T>& pf);
}

#include "mo/convex_hull.cc"