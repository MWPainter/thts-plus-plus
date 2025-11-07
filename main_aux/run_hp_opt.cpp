#include "main_aux/run_xpr.h"

#include "helper_templates.h"

#include "mc_eval.h"

#include "thts.h"
#include "py/py_thts.h"
#include "py/py_multiprocessing_thts_env.h"

#include "py/py_helper.h"
#include <Python.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

using namespace std;
namespace py = pybind11;
using namespace thts;
using namespace thts::python;

namespace thts {

    /**
     * Very coarsely check bayesopt params provided in config
     * Not checked to same standard do in the managers though
     */
    void _validate_bayesopt_params_provided(HpoptConfigMap& xpr_config)
    {
        if (!xpr_config.contains(HPOPT_PARAM_ID_ESTIMATE_CONFIDENCE_THRESHOLD)) 
        {
            throw runtime_error("Expecting estimate confidence threshold specified in xpr config for hp opt.");
        }
        if (!xpr_config.contains(HPOPT_PARAM_ID_BAYESOPT_TOTAL_SAMPLES)) 
        {
            throw runtime_error("Expecting total samples specified in xpr config for hp opt.");
        }
        if (!xpr_config.contains(HPOPT_PARAM_ID_BAYESOPT_INIT_RAND_SAMPLES)) 
        {
            throw runtime_error("Expecting number of initial random samples specified in xpr config for hp opt.");
        }
        if (!xpr_config.contains(HPOPT_PARAM_ID_BAYESOPT_RELEARN_FREQ)) 
        {
            throw runtime_error("Expecting relearn frequence specified in xpr config for hp opt.");
        }
    }

    /**
     * Entry point, hpopt_xpr_id_prefix from command line
     * Checks if any run's need python
     * If so, makes an interpreter and releases gil
     */
    void main_hp_opt(string hpopt_xpr_id_prefix)
    {
        // Read in config
        vector<HpoptConfigMap> hpopt_configs = HpoptManager::lookup_config_vector_from_xpr_prefix(hpopt_xpr_id_prefix);
        _validate_bayesopt_params_provided(hpopt_configs[0]);
        bayesopt::Parameters bo_params = HpoptManager::get_bayesopt_params_from_xpr_config(hpopt_configs[0]);
        vector<HpoptManager> hpopt_managers = HpoptManager::get_hpopt_managers_from_config_vector(hpopt_configs);

        // Check if any run ids need python
        bool need_python = false;
        for (HpoptManager& hpopt_manager : hpopt_managers) {
            if (hpopt_manager.is_python_env()) {
                need_python = true;
                break;
            }
        }

        // If running python, make interpreter and release gil
        shared_ptr<py::scoped_interpreter> py_interpreter;
        shared_ptr<py::gil_scoped_release> release;
        if (need_python) {
            py_interpreter = make_shared<py::scoped_interpreter>();
            release = make_shared<py::gil_scoped_release>();
        }   

        // Actually run optimisation
        for (HpoptManager& hpopt_manager : hpopt_managers) {
            hpopt_manager.open_hpopt_summary_filestream();
            hpopt_manager.write_hpopt_summary_header();

            bayesopt::vectord _results(hpopt_manager->num_hyperparams);
            hpopt_manager->optimize(_results);

            hpopt_manager.write_hpopt_summary_footer();
            hpopt_manager.close_hpopt_summary_filestream();
        }
    }



    HpoptManager::get_bayesopt_params()
    {
        bayesopt::Parameters bo_params;
        bo_params.surr_name = "sGaussianProcessML";
        bo_params.noise = std_mean_eval_threshold*std_mean_eval_threshold; //1.0; 
        bo_params.n_iterations = 190;
        bo_params.n_init_samples = 10;
        bo_params.n_iter_relearn = 10;
        bo_params.verbose_level = 0;
    }

}
