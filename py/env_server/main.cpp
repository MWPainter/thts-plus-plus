/**
 * Entry point for a Python server 
 */

#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include "py/gym_multiprocessing_thts_env.h"
#include "py/mo_gym_multiprocessing_thts_env.h"
#include "py/mo_py_multiprocessing_thts_env.h"
#include "py/py_multiprocessing_thts_env.h"

using namespace std;
namespace py = pybind11;
using namespace thts;
using namespace thts::python;

int main(int argc, char* argv[]) 
{
    py::scoped_interpreter py_interpreter;

    unique_ptr<PyMultiprocessingThtsEnv> thts_env;
    shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();

    string env_type_id = string(argv[1]);
    string thts_unique_filename = string(argv[2]);
    int client_tid = stoi(string(argv[3]));

    if (env_type_id == GYM_ENV_SERVER_ID) 
    {
        string gym_env_id = string(argv[4]);
        thts_env = make_unique<GymMultiprocessingThtsEnv>(pickle_wrapper, thts_unique_filename, gym_env_id, true);
    }
    else if (env_type_id == MOGYM_ENV_SERVER_ID)
    {
        string gym_env_id = string(argv[4]); 
        thts_env = make_unique<MoGymMultiprocessingThtsEnv>(pickle_wrapper, thts_unique_filename, gym_env_id, true);

    }
    else if (env_type_id == PY_ENV_SERVER_ID)
    {
        string module_name = string(argv[4]);
        string class_name = string(argv[5]);
        py::dict kw_args;
        for (int i=6; i<argc; i+=2) {
            kw_args[py::cast(string(argv[i]))] = py::cast(string(argv[i+1]));
        }
        thts_env = make_unique<PyMultiprocessingThtsEnv>(
            pickle_wrapper, thts_unique_filename, module_name, class_name, make_shared<py::dict>(kw_args), true);
    }
    else if (env_type_id == MOPY_ENV_SERVER_ID)
    {
        string module_name = string(argv[4]);
        string class_name = string(argv[5]);
        py::dict kw_args;
        for (int i=6; i<argc; i+=2) {
            kw_args[py::cast(string(argv[i]))] = py::cast(string(argv[i+1]));
        }
        thts_env = make_unique<MoPyMultiprocessingThtsEnv>(
            pickle_wrapper, thts_unique_filename, module_name, class_name, make_shared<py::dict>(kw_args), true);
    }

    thts_env->server_main(client_tid);
}