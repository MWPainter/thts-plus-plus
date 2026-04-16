#pragma once

#include "main_mo/run_manager.h"

#include "mo/mo_mc_eval.h"
#include "mo/algorithms/prior/chvi.h"

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
    MoEvalMetrics run_searches(
        RunManager& run_manager, 
        bool hpopt=false, 
        bool log_trees=true, 
        bool log_convex_hulls=true, 
        int repeats_already_run=0);

    MoEvalMetrics run_chvi(RunManager& run_manager, bool log_convex_hulls=true);


    /**
     * Perform an mc eval and return the MO eval metrics
    */
    MoEvalMetrics run_evals(
        std::shared_ptr<EvalPolicy> eval_policy,
        std::shared_ptr<MoThtsEnv> env, 
        std::shared_ptr<MoThtsManager> thts_manager,
        RunManager& run_manager);

    MoEvalMetrics run_evals_thts(
        std::shared_ptr<MoThtsEnv> env, 
        std::shared_ptr<MoThtsDNode> root_node, 
        std::shared_ptr<MoThtsManager> thts_manager,
        RunManager& run_manager);

    MoEvalMetrics run_evals_chvi(
        std::shared_ptr<MoThtsEnv> env, 
        std::shared_ptr<Chvi> chvi,
        std::shared_ptr<MoThtsManager> thts_manager,
        RunManager& run_manager);
}