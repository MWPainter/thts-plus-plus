#pragma once

#include "main/run_id.h"

#include <ctime>
#include <memory>
#include <vector>

namespace thts {
    /**
     * Performs all of the (replicated) runs corresponding to 'run_id', returning avg expected utility over replicates
    */
    double run_expr(RunID &run_id);

    /**
     * Performs all of the (replicated) runs corresponding to each 'run_id' in 'run_ids'
    */
    void run_exprs(std::shared_ptr<std::vector<RunID>> run_ids);

    /**
     * Performs hyperparameter search corresponding to experiment id 'expr_id'
     */
    void run_hp_opt(std::string expr_id_prefix);

    /**
     * Computes an estimate for the noise parameter of bayesopt for environment with 'env_id' using a random policy 
     */
    void estimate_noise_for_hp_opt(std::string env_id);
}