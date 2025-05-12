#pragma once

#include <unordered_set>
#include <utility>

#include <Eigen/Dense>

#include "mo/mo_helper.h"




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
    class ParetoFront {
        protected:
            std::unordered_set<Eigen::ArrayXd> pf_points;

        public:
            /**
             * Constructor, empty
            */
            ParetoFront();

            /**
             * Constructor, adding points immediately
             * With an option to say if we know that the set of points is already a pareto front
            */
            ParetoFront(const std::unordered_set<Eigen::ArrayXd>& init_points, bool already_pareto_front=false);

            /**
             * Constructor, with a single point
            */
            ParetoFront(const Eigen::ArrayXd& heuristic_val);

            /**
             * Copy constructor
            */
            ParetoFront(const ParetoFront& pf);

            /**
             * Move constructor
            */
            ParetoFront(const ParetoFront&& pf);

        protected:
            // /**
            //  * Adds points to the ParetoFront (i.e. the pf_points member).
            //  * 'points_to_add' doesnt have to form a ParetoFront itself
            //  * This function will prune points appropriately so that the 'pf_points' member will form a ParetoFront
            // */
            // void add_points(const std::unordered_set<TaggedPoint<T>>& points_to_add);

            /**
             * Pareto domination relationship (if u weakly dominates v)
             */
            static bool weakly_pareto_dominates(const Eigen::ArrayXd& u, const Eigen::ArrayXd& v);

            /**
             * Returns the set of points from 'points' that are not (weakly) dominated by any points in 'ref_points'
             * N.B. need to be careful using this. If we have Pareto Fronts U and V which both contain a vector v, then 
             *      v \notin prune(U,V) + prune(V,U)
             *  This happens because we remove v from V in prune(U,V) and remove v from U in prune() 
             * So, if v is in 'ref_points' and 'points' then the returned set will *not* contain v.
            */
            static std::unordered_set<Eigen::ArrayXd> prune(
                const std::unordered_set<Eigen::ArrayXd>& ref_points, 
                const std::unordered_set<Eigen::ArrayXd>& points);

            /**
             * Returns the Pareto front of the set of 'points'.
             * Because we use weak pareto domination, 'prune(points,points)' would return an empty set
            */
            static std::unordered_set<Eigen::ArrayXd> prune(const std::unordered_set<Eigen::ArrayXd>& points);

        public:
            /**
             * Gets the size of the pareto front 
            */
            std::size_t size() const;

            /**
             * Scale ParetoFront by a vector
            */
            ParetoFront scale(double scale) const;

            /**
             * Union of two pareto fronts ('union' is a keyword in c++, so called this combine)
             * If have pfs U and V, then the union is prune({u | u in U or u in V})
            */
            ParetoFront combine(const ParetoFront& other) const;

            /**
             * Add two pareto fronts 
             * If have pfs U and V, then addition is is prune({u+v | u in U, v in V})
             * 
             * The 'tag' member of TaggedPoint is a bit ambiguous with this function. We would only use this in chance 
             * nodes in CHMCTS, where the tag isn't relevant
            */
            ParetoFront add(const ParetoFront& other) const;

            /**
             * Adds a vector to this pareto front
             * If have vector v and pareto front U, then U+v = {u+v | u in U}
            */
            ParetoFront add(const Eigen::ArrayXd& v) const;

            /** 
             * Get points in pareto front
             */
            const std::unordered_set<Eigen::ArrayXd>& get_points() const;

    };
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
    ParetoFront operator*(const ParetoFront pf, double s);
    
    
    ParetoFront operator*(double s, const ParetoFront& pf);

    /**
     * Union of two pareto fronts
    */
    
    ParetoFront operator|(const ParetoFront& pf1, const ParetoFront& pf2);

    /**
     * Sum of pareto fronts
    */
    
    ParetoFront operator+(const ParetoFront& pf1, const ParetoFront& pf2);

    /**
     * Add vector to pareto front
    */
    
    ParetoFront operator+(const ParetoFront& pf, const Eigen::ArrayXd& v);

    
    ParetoFront operator+(const Eigen::ArrayXd& v, const ParetoFront& pf);

    /**
     * Output stream
    */
    
    ostream& operator<<(ostream& os, const ParetoFront& pf);
}
