#include "py/gym_multiprocessing_thts_env.h"

namespace py = pybind11;
using namespace std; 

/**
 * 
 */
namespace thts::python { 

    GymMultiprocessingThtsEnv::GymMultiprocessingThtsEnv(
        shared_ptr<PickleWrapper> pickle_wrapper,
        string& thts_unique_filename,
        string& gym_env_id,
        bool is_server_process) :
            PyMultiprocessingThtsEnv(pickle_wrapper, thts_unique_filename, nullptr, is_server_process),
            gym_env_id(gym_env_id)
    {
        py::gil_scoped_acquire acquire;
        py::module_ py_thts_env_module = py::module_::import("gym_thts_env"); 
        py::object py_thts_env_py_obj = py_thts_env_module.attr("GymThtsEnv")(gym_env_id);
        py_thts_env = make_shared<py::object>(py_thts_env_py_obj);
        _is_fully_observable = py_thts_env->attr("is_fully_observable")().cast<bool>();
    }

    GymMultiprocessingThtsEnv::GymMultiprocessingThtsEnv(GymMultiprocessingThtsEnv& other) :
        PyMultiprocessingThtsEnv(other),
        gym_env_id(other.gym_env_id)
    {
    }

    shared_ptr<ThtsEnv> GymMultiprocessingThtsEnv::clone() {
        return make_shared<GymMultiprocessingThtsEnv>(*this);
    }

    /**
     * Gets the id to identify what type of python multiprocessing env this is
     */
    string GymMultiprocessingThtsEnv::get_multiprocessing_env_type_id() 
    {
        return GYM_ENV_SERVER_ID;
    }

    /**
     * Adds the arguments needed in to run the "py_env_server" program for this env.
     */
    void GymMultiprocessingThtsEnv::fill_multiprocessing_args(vector<string>& args, int tid)
    {
        args.push_back(get_multiprocessing_env_type_id());
        args.push_back(thts_unique_filename);
        args.push_back(to_string(tid));
        args.push_back(gym_env_id);
    }
} 