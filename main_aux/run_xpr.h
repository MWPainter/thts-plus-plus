#pragma once

#include "main_aux/run_manager.h"

#include <ctime>
#include <memory>
#include <vector>

namespace thts {
    /**
     * Main entry point for running experiments
     * Performs all of the (replicated) runs corresponding xpr_id with prefix 'xpr_id_prefix'
     * xpr_dir_override: if non-empty, overrides the experiment directory name (to add results to existing experiment)
    */
    void main_xpr(std::string xpr_id_prefix, std::string xpr_dir_override="", int repeats_already_run=0);

    /**
     * Runs the searches corresponding to a RunManager
     * Returns the final eval_mean of the final search (which is only used in hp opt)
     * Flag to change some parts of the loop when running hpopts
     */
    double run_searches(RunManager& run_manager, bool hpopt=false, bool log_trees=true, int repeats_already_run=0);

    /**
     * Perform an mc eval (of policy from tree node)
    */
    std::pair<double,double> mc_eval(
        std::shared_ptr<ThtsEnv> env, 
        std::shared_ptr<ThtsDNode> root_node, 
        std::shared_ptr<ThtsManager> thts_manager,
        RunManager& run_manager);
}