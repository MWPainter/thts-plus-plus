#include "main/run_id.h"
#include "main/run_expr.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        throw runtime_error("Expecting exactly two arguments: [eval|opt] [expr_id], specifying if we want to run an "
                            "eval experiment, or perform hyperparamter optimisation.");
    }

    if (string(argv[1]) == "eval") {
        shared_ptr<vector<RunID>> run_ids = thts::get_run_ids_from_expr_id(argv[1]);
        thts::run_exprs(run_ids);
    } else if (string(argv[1]) == "opt") {  
        // TODO:
        // #1. Install BayesOpt
        // #2. do example here
        // #3. Clean up run_ids
        // #3.1. work out how to store params in dict in the filenames/folders appropriately - maybe just subdirs?
        // #3.2. Look up max file len in linux? 
        // #4. Clean up plot stuff
        // #4.1. Add helper routine to by default use most recent run of expriment
        // #5. Implement hyperparam opt in run_expr.cpp (and rename it to run.cpp)
    }

    return 0;
}