#include "mo/mo_helper.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "helper_templates.h"
#include "thts_manager.h"

#include <stdexcept>

#include "mo/mo_thts_env.h"
#include "mo/mo_thts_manager.h"

using namespace std;
namespace py = pybind11;


namespace std {

    /**
     * Hash for Eigen::ArrayXd
    */
    size_t hash<Eigen::ArrayXd>::operator()(const Eigen::ArrayXd& v) const {
        size_t cur_hash = 0;
        cur_hash = helper::hash_combine(cur_hash, v.size());
        for (int i=0; i<v.size(); i++) {
            cur_hash = helper::hash_combine(cur_hash, v(i));
        }
        return cur_hash; 
    }

    /**
     * Equals for Eigen::ArrayXd
    */
    bool equal_to<Eigen::ArrayXd>::operator()(const Eigen::ArrayXd& u, const Eigen::ArrayXd& v) const {
        if (u.size() != v.size()) {
            throw runtime_error("Cant compare eigen arrays of different sizes");
        }
        return (u == v).all();
    }
}

namespace thts::helper {
    double norm(const Eigen::ArrayXd& x) {
        return sqrt(x.pow(2.0).sum());
    }

    Eigen::ArrayXd normalised(const Eigen::ArrayXd& x) {
        return x / norm(x);
    }

    Eigen::ArrayXd project(const Eigen::ArrayXd& direction, const Eigen::ArrayXd& x) {
        Eigen::ArrayXd normalised_dir = normalised(direction);
        return normalised_dir * dot(normalised_dir, x);
    }

    double dist(const Eigen::ArrayXd& p1, const Eigen::ArrayXd& p2) {
        return norm(p1-p2);
    }

    double dot(const Eigen::ArrayXd& p1, const Eigen::ArrayXd& p2) {
        return (p1 * p2).sum();
    }

    
    ConstHeuristicFn::ConstHeuristicFn(Eigen::ArrayXd& const_val) : const_val(const_val)
    {
    };
    
    Eigen::ArrayXd ConstHeuristicFn::heuristic_fn(std::shared_ptr<const State> s, MoThtsEnv& env, MoThtsManager& manager, int depth) 
    {
        return const_val;
    }

    /**
     * https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
     */
    Eigen::ArrayXd sample_uniform_random_simplex_vector(RandManager& manager, int dim)
    {
        Eigen::ArrayXd sampled_weight = Eigen::ArrayXd(dim);
        vector<double> exp_rvs_run_sum(dim);
        double exp_rvs_sum = 0.0; 
        for (int i=0; i<dim; i++) {
            double exp_rv = manager.get_rand_exp();
            exp_rvs_sum += exp_rv;
            exp_rvs_run_sum[i] = exp_rvs_sum;
        }
        double prev_run_sum = 0.0;
        for (int i=0; i<dim; i++) {
            sampled_weight[i] = (exp_rvs_run_sum[i] - prev_run_sum) / exp_rvs_sum;
            prev_run_sum = exp_rvs_run_sum[i];
        }
        return sampled_weight;
    } 


    /**
     * Get filename for cached points
     */
    string well_spaced_points_filename(int num_points, int dim, bool is_simplex)
    {
        stringstream ss;
        ss << "util/cached_";
        if (is_simplex) {
            ss << "simplex";
        } else {
            ss << "hypersphere";
        }
        ss << "_points/" << to_string(dim) << "_dim/" << to_string(num_points) << "_points.txt";
        return ss.str();
    }

    /**
     * Generate well spaced points via python script
     */
    void ensure_well_spaced_points_generated(int num_points, int dim, bool is_simplex)
    {
        // If file already exists, we did this work before
        string filename = well_spaced_points_filename(num_points, dim, is_simplex);
        if (filesystem::exists(filename)) {
            return;
        }

        cout << "Generating well spaced points (";
        if (is_simplex) {
            cout << "simplex";
        } else {
            cout << "hypersphere";
        }
        cout  << ", " << dim << "dims, " << num_points 
            << "points) for the first time. This may take a little while." << endl;

        // Ensure a python interpreter exists and gil acquired
        unique_ptr<py::scoped_interpreter> py_interpreter;
        std::unique_ptr<py::gil_scoped_acquire> acquire;
        if (Py_IsInitialized()) {
            acquire = make_unique<py::gil_scoped_acquire>();
        } else {
            py_interpreter = make_unique<py::scoped_interpreter>();
        }

        // Run appropriate python function
        py::module_ py_module = py::module_::import("util.generate_well_spaced_vectors");
        string fn_name = is_simplex ? "generate_and_cache_simplex_points" : "generate_and_cache_hypersphere_points";
        py::object py_gen_and_cache_points_fn = py_module.attr(fn_name.c_str());
        py_gen_and_cache_points_fn(num_points, dim);
        
        cout << "Finished generating well spaced points." << endl;
    }

    /**
     * Makes sure well spaced points are generated
     * Loads them in from the cached file
     */
    vector<Eigen::ArrayXd> get_well_spaced_points(int num_points, int dim, bool is_simplex)
    {
        ensure_well_spaced_points_generated(num_points, dim, is_simplex);
        string filename = well_spaced_points_filename(num_points, dim, is_simplex);

        ifstream cache_file(filename);
        string line;
        vector<Eigen::ArrayXd> vectors;
        vectors.reserve(num_points);
        while (getline(cache_file,line)) {
            vector<string> vec_as_string = thts::helper::string_split(line);
            Eigen::ArrayXd vec(dim);
            for (int i=0; i<dim; i++) {
                vec[i] = stod(vec_as_string[i]);
            }
            vectors.push_back(vec);
        }
        return vectors;
    }

    vector<Eigen::ArrayXd> get_well_spaced_hyperphere_points(int num_points, int dim) {
        return get_well_spaced_points(num_points, dim, false);
    }

    vector<Eigen::ArrayXd> get_well_spaced_simplex_points(int num_points, int dim) {
        return get_well_spaced_points(num_points, dim, true);
    }

    /**
     * Implementation of the rollout heuristic function.
     */
    Eigen::ArrayXd mo_rollout_heuristic_fn(
        shared_ptr<const State> state, ThtsEnv& env, ThtsManager& manager, int depth) 
    {
        MoThtsEnv& mo_env = dynamic_cast<MoThtsEnv&>(env);
        MoThtsManager& mo_manager = (MoThtsManager&) manager;
        ThtsContext& ctx = *manager.get_thts_context();
        int rollout_steps_left = manager.max_depth - depth;
        Eigen::ArrayXd rollout_reward = Eigen::ArrayXd::Zero(mo_manager.reward_dim);

        while (rollout_steps_left-- > 0 && !mo_env.is_sink_state_itfc(state, ctx)) {
            shared_ptr<ActionVector> actions = mo_env.get_valid_actions_itfc(state, ctx);
            int index = manager.get_rand_int(0, actions->size());
            shared_ptr<const Action> action = actions->at(index);
            rollout_reward += mo_env.get_mo_reward_itfc(state, action, ctx);
            state = env.sample_transition_distribution_itfc(state, action, manager, ctx);
        }
        
        return rollout_reward;
    }
}