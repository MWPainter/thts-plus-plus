#pragma once

#include "main_aux/run_manager.h"

#include <ctime>
#include <memory>
#include <vector>

namespace thts {
    /**
     * Main entry point for running experiments
     * Performs all of the (replicated) runs corresponding xpr_id with prefix 'xpr_id_prefix'
    */
    void main_xpr(std::string xpr_id_prefix);

}