#include "py/mo_gym_multiprocessing_thts_env.h"

namespace py = pybind11;
using namespace std; 

/**
 * 
 */
namespace thts::python { 

    MoGymMultiprocessingThtsEnv::MoGymMultiprocessingThtsEnv(
        shared_ptr<PickleWrapper> pickle_wrapper,
        string& gym_env_id) :
            ThtsEnv(true),
            MoPyMultiprocessingThtsEnv(pickle_wrapper, nullptr),
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
        MoPyMultiprocessingThtsEnv(other.pickle_wrapper, other.py_thts_env),
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
} 