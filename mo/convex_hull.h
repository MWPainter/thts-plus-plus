
#pragma once

#include "mo/pareto_front.h"

#include <unordered_set>
#include <utility>

#include <Eigen/Dense>

namespace thts {
    /**     
     * Convex Hull implementation
     * 
     * As a convex hull can be considered a Pareto front when using mixed policies and linear scalarisations, we make 
     * it a subclass of Pareto Front (i.e. a CH is a PF, but a PF isn't necessarily a CH)
     * 
     * TODO: 
     *  - Would like to make this a subclass of Pareto Front (because a Convex Hull is a Pareto front when using mixed
     *      policies and linear scalarisations).
     *  - Also ParetoFront calls 'prune' and need to make that virtual between the CH and PF, but cant call virt funcitons 
     *      from constructor and dont want to sort that out now
     *  - So for now, we're just going to accept code duplication :(
     * 
     * TODO: use description have in local branch, copy implementations from here
     * TODO: add comments for functions that have different imlpementations to pareto fronts
     * TODO: also add comments saying extra things that we could implement
     *      - use qhul convex hull (adding a reference point)
     *      - computing the hypervolume using convex hull (will actually need to do this for hypervolume indicator act selection)
     * 
     * Member variables:
     *      ch_points: 
     *          The set of TaggedPoints in the Convex Hull          
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
            ConvexHull(const Eigen::ArrayXd& heuristic_val, const T& tag);
            ConvexHull(const ConvexHull<T>& ch);
            ConvexHull(const ConvexHull<T>&& ch);

            /**
             * Assignment operators
            */
            ConvexHull<T>& operator=(const ConvexHull<T>& ch);
            ConvexHull<T>& operator=(ConvexHull<T>&& ch);
            inline ConvexHull<T>& operator*=(double rhs);
            inline ConvexHull<T>& operator|=(ConvexHull<T>& rhs);
            inline ConvexHull<T>& operator|=(ConvexHull<T>&& rhs);
            inline ConvexHull<T>& operator+=(ConvexHull<T>& rhs);
            inline ConvexHull<T>& operator+=(ConvexHull<T>&& rhs);
            inline ConvexHull<T>& operator+=(Eigen::ArrayXd& rhs);

        protected:
            /**
             * Returns if 'point' is dominated by any ppints in 'ref_points', which is checked using a linear program
             * This is a strong domination compared to the weak notion we used in ParetoFront
             * 
             * If ignore_if_point_in_ref_points then we use ref_points-{point} in place of ref_points
            */
            bool strongly_convex_dominated(
                const std::unordered_set<TaggedPoint<T>>& ref_points, 
                const TaggedPoint<T>& point) const;
            /**
             * Main functions that are overriden from Pareto Front
            */
            // std::unordered_set<TaggedPoint<T>> prune(
            //     const std::unordered_set<TaggedPoint<T>>& ref_points, 
            //     const std::unordered_set<TaggedPoint<T>>& points) const;
            std::unordered_set<TaggedPoint<T>> prune(const std::unordered_set<TaggedPoint<T>>& points) const;

        public:
            std::size_t size() const;
            void set_tags(const T& new_tag);
            ConvexHull<T> scale(double scale) const;
            ConvexHull<T> combine(const ConvexHull<T>& other) const;
            ConvexHull<T> add(const ConvexHull<T>& other) const;
            ConvexHull<T> add(const Eigen::ArrayXd& v) const;

            /**
             * TODO: want this directly implemented in operator<<
             * But declaring operator<< as friend wasnt working because I couldnt work out how to declare a templated 
             * function as a friend hmph
            */
            void write_to_ostream(std::ostream& os) const;
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
    ConvexHull<T> operator*(const ConvexHull<T>& ch, double s);
    
    template <typename T>
    ConvexHull<T> operator*(double s, const ConvexHull<T>& ch);

    /**
     * Union of two pareto fronts
    */
    template <typename T>
    ConvexHull<T> operator|(const ConvexHull<T>& ch1, const ConvexHull<T>& ch2);

    /**
     * Sum of pareto fronts
    */
    template <typename T>
    ConvexHull<T> operator+(const ConvexHull<T>& ch1, const ConvexHull<T>& ch2);

    /**
     * Add vector to pareto front
    */
    template <typename T>
    ConvexHull<T> operator+(const ConvexHull<T>& ch, const Eigen::ArrayXd& v);

    template <typename T>
    ConvexHull<T> operator+(const Eigen::ArrayXd& v, const ConvexHull<T>& ch);

    /**
     * Output stream
    */
    template <typename T>
    ostream& operator<<(ostream& os, const ConvexHull<T>& ch);
}

#include "mo/convex_hull.cc"