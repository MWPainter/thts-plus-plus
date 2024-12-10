#include "py/mo_gym_multiprocessing_thts_env.h"

namespace py = pybind11;
using namespace std; 

/**
 * 
 */
namespace thts::python { 

    MoGymMultiprocessingThtsEnv::MoGymMultiprocessingThtsEnv(
        shared_ptr<PickleWrapper> pickle_wrapper,
        string& thts_unique_filename,
        string& gym_env_id,
        bool is_server_process) :
            ThtsEnv(true),
            MoPyMultiprocessingThtsEnv(pickle_wrapper, thts_unique_filename, nullptr, is_server_process),
            gym_env_id(gym_env_id)
    {
        py::gil_scoped_acquire acquire;
        py::module_ py_thts_env_module = py::module_::import("mo_gym_thts_env"); 
        py::object py_thts_env_py_obj = py_thts_env_module.attr("MoGymThtsEnv")(gym_env_id);
        py_thts_env = make_shared<py::object>(py_thts_env_py_obj);
        ThtsEnv::_is_fully_observable = py_thts_env->attr("is_fully_observable")().cast<bool>();
        reward_dim = py_thts_env->attr("get_reward_dim")().cast<int>();
    }

    MoGymMultiprocessingThtsEnv::MoGymMultiprocessingThtsEnv(MoGymMultiprocessingThtsEnv& other) :
        ThtsEnv(other.ThtsEnv::_is_fully_observable),
        MoPyMultiprocessingThtsEnv(other),
        gym_env_id(other.gym_env_id)
    {
    }

    shared_ptr<ThtsEnv> MoGymMultiprocessingThtsEnv::clone() {
        return make_shared<MoGymMultiprocessingThtsEnv>(*this);
    }

    string MoGymMultiprocessingThtsEnv::get_multiprocessing_env_type_id() 
    {
        return MOGYM_ENV_SERVER_ID;
    }

    /**
     * Adds the arguments needed in to run the "py_env_server" program for this env.
     */
    void MoGymMultiprocessingThtsEnv::fill_multiprocessing_args(vector<string>& args, int tid)
    {
        args.push_back(get_multiprocessing_env_type_id());
        args.push_back(thts_unique_filename);
        args.push_back(to_string(tid));
        args.push_back(gym_env_id);
    }
} 