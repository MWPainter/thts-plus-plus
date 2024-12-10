#include "py/timed_mo_gym_multiprocessing_thts_env.h"

#include <iostream>

namespace py = pybind11;
using namespace std; 

/**
 * 
 */
namespace thts::python { 

    /**
     * Add one to reward dim, as we're adding a reward
     */
    TimedMoGymMultiprocessingThtsEnv::TimedMoGymMultiprocessingThtsEnv(
        shared_ptr<PickleWrapper> pickle_wrapper,
        string& thts_unique_filename,
        string& gym_env_id,
        bool is_server_process) :
            MoGymMultiprocessingThtsEnv(pickle_wrapper, thts_unique_filename, gym_env_id)
    {
        py::gil_scoped_acquire acquire;
        reward_dim = py_thts_env->attr("get_reward_dim")().cast<int>() + 1;
    }

    /**
     * Add one to reward dim, as we're adding a reward
     */
    TimedMoGymMultiprocessingThtsEnv::TimedMoGymMultiprocessingThtsEnv(TimedMoGymMultiprocessingThtsEnv& other) :
        MoGymMultiprocessingThtsEnv(other)
    {
        py::gil_scoped_acquire acquire;
        reward_dim = py_thts_env->attr("get_reward_dim")().cast<int>() + 1;
    }

    shared_ptr<ThtsEnv> TimedMoGymMultiprocessingThtsEnv::clone() {
        return make_shared<TimedMoGymMultiprocessingThtsEnv>(*this);
    }

    /**
     * As adding time cost client side, dont need server to be aware of the additional reward, and no need code bloat
     */
    string TimedMoGymMultiprocessingThtsEnv::get_multiprocessing_env_type_id() 
    {
        return MOGYM_ENV_SERVER_ID;
    }

    /**
     * Add time cost
     * 
     * TODO: sure I can extend this array somehow rather than making new and copying out, but eigen website down when 
     *      writing this.
     */
    Eigen::ArrayXd TimedMoGymMultiprocessingThtsEnv::get_mo_reward(
        shared_ptr<const PyState> state, 
        shared_ptr<const PyAction> action,
        ThtsEnvContext& ctx) const 
    {
        shared_mem_wrapper->rpc_id = RPC_get_reward;
        shared_mem_wrapper->value_type = SMT_strings;
        shared_mem_wrapper->strings = make_shared<vector<string>>();
        shared_mem_wrapper->strings->push_back(*state->get_serialised_state());
        shared_mem_wrapper->strings->push_back(*action->get_serialised_action());
        shared_mem_wrapper->make_rpc_call();

        Eigen::ArrayXd mo_reward = Eigen::ArrayXd::Zero(reward_dim);
        for (size_t i=0; i<shared_mem_wrapper->doubles->size(); i++) {
            mo_reward[i] = shared_mem_wrapper->doubles->at(i);
        }

        mo_reward[reward_dim-1] = -1.0;

        return mo_reward;
    }
} 