#pragma once

#include "main_aux/hpopt_manager.h"

#include <ctime>
#include <memory>
#include <vector>

namespace thts {

    /**
     * Main entry point for running hyperparameter optimisation
     * Performs hp opt corresponding xpr_id with prefix 'xpr_id_prefix'
     */
    void main_hp_opt(std::string hpopt_xpr_id_prefix);
}