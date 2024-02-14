
// // TODO: make convex hull bit
// // TODO: make convex hull subclass of pareto front (as it is a pareto front with mixed policies)
// // TODO: make 2D convex hull - which implements the value curve backend
// // TODO: use linear program lib for convex hull pruning: https://www.alglib.net/download.php
// // https://github.com/ori-goals/rapport-algorithms/blob/mike-thts/rapport_algorithms/thts/utils/pareto_front.py

// // qhull: https://pkg.cs.ovgu.de/LNF/i386/5.11/LNFqhull-docs/reloc/libqhull/qh-code.htm#qh-cpp


// #pragma once

// #include "mo/pareto_front.h"


// namespace thts {
//     /**
//      * Convex Hull implementation
//      * 
//      * As a convex hull can be considered a Pareto front when using mixed policies and linear scalarisations, we make 
//      * it a subclass of Pareto Front (i.e. a CH is a PF, but a PF isn't necessarily a CH)
//      * 
//      * 
//     */
//     template <typename T>
//     class ConvexHull  {

//         public:
//             /**
//              * Duplicate all the PF constructors
//             */
//             ConvexHull();
//             ConvexHull(const std::vector<std::pair<Eigen::ArrayXd,T>>& init_points);
//             ConvexHull(const std::vector<Eigen::ArrayXd>& init_points, const T& tag);
//             ConvexHull(const std::unordered_set<TaggedPoint<T>>& init_points, bool already_convex_hull=false);
//             ConvexHull(const ConvexHull<T>& pf);
//             ConvexHull(const ConvexHull<T>&& pf);
        
//         private:
//             /**
//              * Constructors that take a pareto front
//             */

//     };
// }

// #include "mo/convex_hull.cc"




#pragma once

#include "mo/pareto_front.h"

#include <unordered_set>
#include <utility>

#include <Eigen/Dense>


namespace thts {
    /**
     * Convex Hull implementation
     * 
     * 
     * TODO: 
     *  - Would like to make this a subclass of Pareto Front (because a Convex Hull is a Pareto front when using mixed
     *      policies and linear scalarisations).
     *  - Also ParetoFront calls 'prune' and need to make that virtual between the CH and PF, but cant call virt funcitons 
     *      from constructor and dont want to sort that out now
     *  - So for now, we're just going to accept code duplication :(
     * 
     * Member variables:
     *      pf_points: 
     *          The set of TaggedPoints in the Pareto Front
     *          
    */
    template <typename T>
    class ParetoFront {
        protected:
            std::unordered_set<TaggedPoint<T>> pf_points;

        public:
            /**
             * Constructor, empty
            */
            ParetoFront();

            /**
             * Constructor, adding points immediately
            */
            ParetoFront(const std::vector<std::pair<Eigen::ArrayXd,T>>& init_points);

            /**
             * Constructor, adding points immediately, with one tag
            */
            ParetoFront(const std::vector<Eigen::ArrayXd>& init_points, const T& tag);

            /**
             * Constructor, set of Tagged points
             * With an option to say if we know that the set of points is already a pareto front
            */
            ParetoFront(const std::unordered_set<TaggedPoint<T>>& init_points, bool already_pareto_front=false);

            /**
             * Copy constructor
            */
            ParetoFront(const ParetoFront<T>& pf);

            /**
             * Move constructor
            */
            ParetoFront(const ParetoFront<T>&& pf);

        protected:
            // /**
            //  * Adds points to the ParetoFront (i.e. the pf_points member).
            //  * 'points_to_add' doesnt have to form a ParetoFront itself
            //  * This function will prune points appropriately so that the 'pf_points' member will form a ParetoFront
            // */
            // void add_points(const std::unordered_set<TaggedPoint<T>>& points_to_add);

            /**
             * Returns the set of points from 'points' that are not (weakly) dominated by any points in 'ref_points'
             * N.B. need to be careful using this. If we have Pareto Fronts U and V which both contain a vector v, then 
             *      v \notin prune(U,V) + prune(V,U)
             *  This happens because we remove v from V in prune(U,V) and remove v from U in prune() 
             * So, if v is in 'ref_points' and 'points' then the returned set will *not* contain v.
            */
            std::unordered_set<TaggedPoint<T>> prune(
                const std::unordered_set<TaggedPoint<T>>& ref_points, 
                const std::unordered_set<TaggedPoint<T>>& points) const;

            /**
             * Returns the Pareto front of the set of 'points'.
             * Because we use weak pareto domination, 'prune(points,points)' would return an empty set
            */
            std::unordered_set<TaggedPoint<T>> prune(const std::unordered_set<TaggedPoint<T>>& points) const;

        public:
            /**
             * Gets the size of the pareto front 
            */
            std::size_t size() const;

            /**
             * Sets the tag of every point in 'pf_points' to have the tag 'new_tag'.
            */
            void set_tags(const T& new_tag);

            /**
             * Scale ParetoFront by a vector
            */
            ParetoFront<T> scale(double scale) const;

            /**
             * Union of two pareto fronts ('union' is a keyword in c++, so called this combine)
             * If have pfs U and V, then the union is prune({u | u in U or u in V})
            */
            ParetoFront<T> combine(const ParetoFront<T>& other) const;

            /**
             * Add two pareto fronts 
             * If have pfs U and V, then addition is is prune({u+v | u in U, v in V})
             * 
             * The 'tag' member of TaggedPoint is a bit ambiguous with this function. We would only use this in chance 
             * nodes in CHMCTS, where the tag isn't relevant
            */
            ParetoFront<T> add(const ParetoFront<T>& other) const;

            /**
             * Adds a vector to this pareto front
             * If have vector v and pareto front U, then U+v = {u+v | u in U}
            */
            ParetoFront<T> add(const Eigen::ArrayXd& v) const;

    };
}

/**
 * Forward declare hash, equality and output stream function sepcialisations for TaggedPoints
 * Hash and equals_to needed so that tagged points can be used in unordered_set
 * Output stream for debugging
*/
namespace std {
    using namespace thts;

    /**
     * Hash
    */
    template <typename T>
    struct hash<TaggedPoint<T>> {
        size_t operator()(const TaggedPoint<T>&) const;
    };

    /**
     * Equals
    */
    template <typename T>
    struct equal_to<TaggedPoint<T>> {
        bool operator()(const TaggedPoint<T>&, const TaggedPoint<T>&) const;
    };

    /**
     * Output stream
    */
    template <typename T>
    ostream& operator<<(ostream& os, const TaggedPoint<T>& point);
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
    ParetoFront<T> operator*(const ParetoFront<T>& pf, double s);
    
    template <typename T>
    ParetoFront<T> operator*(double s, const ParetoFront<T>& pf);

    /**
     * Union of two pareto fronts
    */
    template <typename T>
    ParetoFront<T> operator%(const ParetoFront<T>& pf1, const ParetoFront<T>& pf2);

    /**
     * Sum of pareto fronts
    */
    template <typename T>
    ParetoFront<T> operator+(const ParetoFront<T>& pf1, const ParetoFront<T>& pf2);

    /**
     * Add vector to pareto front
    */
    template <typename T>
    ParetoFront<T> operator+(const ParetoFront<T>& pf, const Eigen::ArrayXd& v);

    template <typename T>
    ParetoFront<T> operator+(const Eigen::ArrayXd& v, const ParetoFront<T>& pf);

    /**
     * Output stream
    */
    template <typename T>
    ostream& operator<<(ostream& os, const ParetoFront<T>& pf);
}

#include "mo/convex_hull.cc"