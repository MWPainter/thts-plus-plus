
#pragma once

#include "mo/data_structures/pareto_front.h"

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
    class ConvexHull {
        // making ch_points public to be able to iterate over them in Pareto UCB
        // long term wouldn't really want that
        // should write a custom iterator (https://internalpointers.com/post/writing-custom-iterators-modern-cpp)
        // but dont have time for that right now
        // protected:
        public: 
            std::unordered_set<Vec> ch_points;

        public:
            ConvexHull();
            ConvexHull(const std::unordered_set<Eigen::ArrayXd>& init_points, bool already_pareto_front=false);
            ConvexHull(const std::unordered_set<Vec>& init_points, bool already_pareto_front=false);
            ConvexHull(const Vec& heuristic_val);
            ConvexHull(const ConvexHull& ch);
            ConvexHull(const ConvexHull&& ch);

            /**
             * Assignment operators
            */
            ConvexHull& operator=(const ConvexHull& ch);
            ConvexHull& operator=(const ConvexHull&& ch);
            ConvexHull& operator*=(double rhs);
            ConvexHull& operator*=(const Vec& rhs);
            ConvexHull& operator|=(const ConvexHull& rhs);
            ConvexHull& operator|=(const ConvexHull&& rhs);
            ConvexHull& operator+=(const ConvexHull& rhs);
            ConvexHull& operator+=(const ConvexHull&& rhs);
            ConvexHull& operator+=(const Vec& rhs);
            ConvexHull& operator-=(const Vec& rhs);

        protected:
            /**
             * Returns if 'point' is dominated by any ppints in 'ref_points', which is checked using a linear program
             * This is a strong domination compared to the weak notion we used in ParetoFront
             * 
             * If ignore_if_point_in_ref_points then we use ref_points-{point} in place of ref_points
            */
            static bool strongly_convex_dominated(
                const std::unordered_set<Vec>& ref_points, 
                const Vec& point);
            /**
             * Main functions that are overriden from Pareto Front
            */
            // std::unordered_set<Vec> prune(
            //     const std::unordered_set<Vec>& ref_points, 
            //     const std::unordered_set<Vec>& points) const;
            static std::unordered_set<Vec> prune(const std::unordered_set<Vec>& points);

        public:
            int reward_dim() const;
            std::size_t size() const;
            ConvexHull scale(double scale) const;
            ConvexHull scale(const Vec& scale) const;
            ConvexHull combine(const ConvexHull& other) const;
            ConvexHull add(const ConvexHull& other) const;
            ConvexHull add(const Vec& v) const;
            ConvexHull subtract(const Vec& v) const;

            /**
             * If this convex hull is equal to another convex hull (ignoring any tags)
             */
            bool equals(const ConvexHull& other) const;

            /**
             * Get the best tag for a context weight
            */
            Vec get_best_point(const Vec& context_weight, RandManager& rand_manager) const;

            /**
             * Get max (linear) utility from this convex hull
             */
            double get_max_linear_utility(const Vec& context_weight) const;

            /**
             * Get the hypervolume of this convex hull
             * ref_point is a reference point that must be weakly dominated by all points in the convex hull
             */
            double hypervolume(const Vec& ref_point) const;

            /**
             * Get the sparsity metric from this convex hull
             * Equation (18) in: https://arxiv.org/pdf/2103.09568
             */
            double sparsity_metric() const;

            /**
             * Get the additive epsilon metric from this convex hull
             * Equation (19) in: https://arxiv.org/pdf/2103.09568
             */
            double additive_eps_metric() const;

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
    ConvexHull operator*(const ConvexHull& ch, double s);
    ConvexHull operator*(double s, const ConvexHull& ch);
    ConvexHull operator*(const ConvexHull& ch, const Vec& v);
    ConvexHull operator*(const Vec& v, const ConvexHull& ch);

    /**
     * Union of two convex hulls
    */
    ConvexHull operator|(const ConvexHull& ch1, const ConvexHull& ch2);

    /**
     * Sum of convex hulls
    */
    ConvexHull operator+(const ConvexHull& ch1, const ConvexHull& ch2);

    /**
     * Add vector to convex hull
    */
    ConvexHull operator+(const ConvexHull& ch, const Vec& v);
    ConvexHull operator+(const Vec& v, const ConvexHull& ch);
    ConvexHull operator-(const ConvexHull& ch, const Vec& v);
    ConvexHull operator-(const Vec& v, const ConvexHull& ch);

    /**
     * Equality of convex hulls
     */
    bool operator==(const ConvexHull& lhs, const ConvexHull& rhs);

    /**
     * Output stream
    */
    ostream& operator<<(ostream& os, const ConvexHull& ch);
}