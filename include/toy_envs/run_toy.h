#pragma once

#include "toy_envs/run_id.h"

namespace thts {
    /**
     * Performs all of the (replicated) runs corresponding to 'run_id'
    */
    void perform_toy_env_runs(RunID& run_id);
}