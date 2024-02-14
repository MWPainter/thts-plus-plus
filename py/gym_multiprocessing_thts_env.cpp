#include "py/gym_multiprocessing_thts_env.h"

namespace py = pybind11;
using namespace std; 

/**
 * 
 */
namespace thts::python { 

    GymMultiprocessingThtsEnv::GymMultiprocessingThtsEnv(
        shared_ptr<PickleWrapper> pickle_wrapper,
        string& gym_env_id) :
            PyMultiprocessingThtsEnv(pickle_wrapper, nullptr),
            gym_env_id(gym_env_id)
    {
        py::module_ py_thts_env_module = py::module_::import("gym_thts_env"); 
        py::object py_thts_env_py_obj = py_thts_env_module.attr("GymThtsEnv")(gym_env_id);
        py_thts_env = make_shared<py::object>(py_thts_env_py_obj);
        _is_fully_observable = py_thts_env->attr("is_fully_observable")().cast<bool>();
    }

    GymMultiprocessingThtsEnv::GymMultiprocessingThtsEnv(GymMultiprocessingThtsEnv& other) :
        PyMultiprocessingThtsEnv(other.pickle_wrapper, other.py_thts_env),
        gym_env_id(other.gym_env_id)
    {
    }

    shared_ptr<ThtsEnv> GymMultiprocessingThtsEnv::clone() {
        return make_shared<GymMultiprocessingThtsEnv>(*this);
    }
} 