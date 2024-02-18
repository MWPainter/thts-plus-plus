#pragma once

#include <unordered_set>
#include <utility>

#include <Eigen/Dense>

namespace thts {
    /**
     * A Point in a Pareto Front / Convex Hull
     * 
     * In practise, we want points to be 'tagged'. For example, it will be useful to tag points with the actions that were 
     * used to obtain that value.
     * 
     * Note that we only really want to use this in ParetoFront's and ConvexHull's. 
     * TODO: consider moving the struct definition into the ParetoFront declaration
     * 
     * N.B. mark tag as mutable, as equality/hash will only depend on 'point'. So it's valid to edit the tag when in an 
     * unordered_set/unordered_map
     * 
     * Member variables:
     *      point: A vector value (i.e. the point)
     *      tag: A tag associated with this point
    */
    template <typename T>
    struct TaggedPoint {
        Eigen::ArrayXd point;
        mutable T tag;
        
        /**
         * Constructor
        */
        TaggedPoint(const Eigen::ArrayXd& point, const T& tag);

        /**
         * Copy constructor
        */
        TaggedPoint(const TaggedPoint<T>& other);

        /**
         * Move constructor
        */
        TaggedPoint(const TaggedPoint<T>&& other);

        /**
         * Copy asignment operator
        */
        TaggedPoint<T>& operator=(const TaggedPoint<T>& other);

        /**
         * Move asignment operator
        */
        TaggedPoint<T>& operator=(const TaggedPoint<T>&& other);

        /**
         * Returns if this TaggedPoint weakly Pareto dominates another TaggedPoint 'other'
         * Technically vector u (strongly) pareto dominates vector v 
         *      iff for all i. u[i]>=v[i] and there exists j. u[j]>v[j]
         * Say u weakly Pareto dominates v 
         *      iff for all i. u[i] >= v[i]
         * As we won't really care much for keeping two vectors with the same values in our algorithms, we'll work with 
         *      weak pareto domination
        */
        bool weakly_pareto_dominates(const TaggedPoint<T>& other) const;

        /**
         * Returns if this TaggedPoint is equal to another TaggedPoint 'other'
         * N.B. this ignores the values of the 'tag' members, it only compares equality for the vecxtors
        */
        bool equals(const TaggedPoint<T>& other) const;

        /**
         * Equality operator
        */
        bool operator==(const TaggedPoint<T>& other) const;

        /**
         * Returns a hash for this tagged point
         * N.B. ignores the value of 'tag' members, as if a == b, then need hash(a) == hash(b)
        */
        std::size_t hash() const;
    };
}

namespace thts {
    /**
     * Pareto Front implementation
     * 
     * Given a set of points, a Pareto Front is the set of points that are not Pareto dominated by any other point in 
     * the set. A vector u Pareto Dominates vector v iff for all i. u[i]>=v[i] and there exists j. u[j]>v[j]. 
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
             * Constructor, with a single point
            */
            ParetoFront(const Eigen::ArrayXd& heuristic_val, const T& tag);

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
    ParetoFront<T> operator|(const ParetoFront<T>& pf1, const ParetoFront<T>& pf2);

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

#include "mo/pareto_front.cc"