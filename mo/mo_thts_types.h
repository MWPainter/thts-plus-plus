#pragma once

#include "thts_types.h"

#include <memory>
#include <Eigen/Dense>

/**
 * thts_types.h
 * 
 * This file contains some base types for multi objective algorithms
 * 
 * For now this should just be some typedefs
 */

namespace thts {


    /**
     * An implementation of state containing a vector of integers as the state.
     */
    class IntVectorState : public State {
        public:
            std::vector<int> state;

            IntVectorState(std::vector<int>& v) : state(v) {}
            virtual ~IntVectorState() = default;
            virtual std::size_t hash() const override;
            bool equals(const IntVectorState& other) const;
            virtual bool equals_itfc(const Observation& other) const override;
            virtual std::string get_pretty_print_string() const override;
    };


    typedef std::unordered_map<std::shared_ptr<const IntVectorState>,double> IntVectorStateDistr;



    /**
     * Typedef for heuristic function pointers
     * First used in thts_decision_node.h and thts_chance_node.h
     * N.B. The & here is to get address as we want function pointers
     */  
    Eigen::ArrayXd _DummyMoHeuristicFn(std::shared_ptr<const State> s, std::shared_ptr<ThtsEnv> env);
    typedef decltype(&_DummyMoHeuristicFn) MoHeuristicFnPtr;





    /**
     * Wrapper around Eigen::ArrayXd to make it hashable and comparable
     * Also makes most operations operate at a vector level
     * For example, if u,v, are Eigen::ArrayXd, then u==v will give a vector of bools, whereas if Vec, then gives a bool
     * Also provides open access to the underlying Eigen::ArrayXd to use Eigen functions where needed
     */
    struct Vec {
        public:
            Eigen::ArrayXd vec;

            Vec(const Eigen::ArrayXd& v);
            Vec(Eigen::ArrayXd&& v);
            Vec(const std::vector<double>& v);
            Vec(const Vec& other);
            Vec(Vec&& other);
            Vec(int dim, float val=0.0);

            double norm() const;
            Vec normalised() const;
            double dot(const Vec& other) const;
            Vec project_onto_origin_line(const Vec& direction) const;
            Vec project_onto_line(const Vec& direction, const Vec& point) const;
            double dist(const Vec& other) const;

            bool weakly_pareto_dominates(const Vec& other) const;

            bool equals(const Vec& other) const;
            std::size_t hash() const;

            Vec operator+(const Vec& other) const;
            Vec operator-(const Vec& other) const;
            Vec operator*(const Vec& other) const;
            Vec operator/(const Vec& other) const;

            Vec& operator=(const Vec& other);
            Vec& operator=(Vec&& other);
            Vec& operator+=(const Vec& other);
            Vec& operator-=(const Vec& other);
            Vec& operator*=(const Vec& other);
            Vec& operator/=(const Vec& other);

            bool operator==(const Vec& other) const;
            bool operator!=(const Vec& other) const;

            double operator[](size_t i) const;
    };
}


namespace std {
    using namespace thts;

    /**
     * IntVectorState hash
    */
    ostream& operator<<(ostream& os, const IntVectorState& state);
    ostream& operator<<(ostream& os, const shared_ptr<const IntVectorState>& state);



    /**
     * Vec Hash
    */
    template <>
    struct hash<Vec> {
        size_t operator()(const Vec&) const;
    };

    /**
     * Vec Equals
    */
    template <>
    struct equal_to<Vec> {
        bool operator()(const Vec& lhs, const Vec& rhs) const;
    };

    /**
     * Vec Output stream
    */
    ostream& operator<<(ostream& os, const Vec& point);


    /**
     * Vec operators with scalars
     */
    Vec operator*(const Vec& v, double s);
    Vec operator*(double s, const Vec& v);
    Vec operator+(const Vec& v, double s);
    Vec operator+(double s, const Vec& v);
    Vec operator-(const Vec& v, double s);
    Vec operator-(double s, const Vec& v);
    Vec operator/(const Vec& v, double s);
    Vec operator/(double s, const Vec& v);

}