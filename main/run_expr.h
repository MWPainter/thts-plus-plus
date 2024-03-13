#pragma once

#include "main/run_id.h"

#include <memory>
#include <vector>

namespace thts {
    /**
     * Performs all of the (replicated) runs corresponding to 'run_id'
    */
    void run_exprs(std::shared_ptr<std::vector<RunID>> run_ids);
}