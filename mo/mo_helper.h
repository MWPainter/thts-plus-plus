#pragma once

#include <Eigen/Dense>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include "thts_types.h"
#include "mo/mo_thts_types.h"
#include "thts_env.h"
#include "thts_manager.h"

/**
 * Hash and equal_to for Eigen::ArrayXd
*/
namespace std {
    // using namespace thts;

    /**
     * Hash
    */
    template <>
    struct hash<Eigen::ArrayXd> {
        size_t operator()(const Eigen::ArrayXd&) const;
    };

    /**
     * Equals
    */
    template <>
    struct equal_to<Eigen::ArrayXd> {
        bool operator()(const Eigen::ArrayXd&, const Eigen::ArrayXd&) const;
    };
}

/**
 * Vector arithmatic helpers
*/
namespace thts::helper {
    double norm(const Eigen::ArrayXd& x);
    Eigen::ArrayXd normalised(const Eigen::ArrayXd& x);
    Eigen::ArrayXd project(const Eigen::ArrayXd& direction, const Eigen::ArrayXd& x);
    double dist(const Eigen::ArrayXd& p1, const Eigen::ArrayXd& p2);
    double dot(const Eigen::ArrayXd& p1, const Eigen::ArrayXd& p2);

    struct ConstHeuristicFn {
        Eigen::ArrayXd const_val;
        ConstHeuristicFn(Eigen::ArrayXd& const_val);
        Eigen::ArrayXd heuristic_fn(std::shared_ptr<const State> s, MoThtsEnv& env, MoThtsManager& manager, int depth);
    };
        
    Eigen::ArrayXd sample_uniform_random_simplex_vector(RandManager& manager, int dim);

    /**
     * Generate well spaced points
     * - interface to python functions to generate well spaced points on a hypersphere/simplex
     * - handles if python interpreter is initialised or not (temporarily initialises an interpreter if not)
     * - assumes that thread calling this function has the GIL, OR, is the only thread running python code at the moment
     */
    std::vector<Eigen::ArrayXd> get_well_spaced_points(size_t num_points, size_t dim, bool is_simplex=false);
    std::vector<Eigen::ArrayXd> get_well_spaced_hyperphere_points(size_t num_points, size_t dim);
    std::vector<Eigen::ArrayXd> get_well_spaced_simplex_points(size_t num_points, size_t dim);

    /**
     * The multi objective rollout heuristic function, that returns an MC estimate of 'state' with a rollout with random policy
     */
    Eigen::ArrayXd mo_rollout_heuristic_fn(
        std::shared_ptr<const State> state, MoThtsEnv& env, MoThtsManager& manager, int depth);







    //v1TODO: helper functions
    string well_spaced_points_filename(int num_points, int dim, bool is_simplex);
    void ensure_well_spaced_points_generated(int num_points, int dim, bool is_simplex);
}