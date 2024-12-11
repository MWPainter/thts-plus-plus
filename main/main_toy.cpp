#include "toy_envs/run_toy.h"
#include "toy_envs/run_id.h"

#include <memory>
#include <stdexcept>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        throw runtime_error("Expecting exactly one argument specifying valid expr id");
    }

    shared_ptr<vector<RunID>> run_ids = thts::get_run_ids_from_expr_id(argv[1]);
    for (RunID run_id : *run_ids) {
        thts::perform_toy_env_runs(run_id);
    }

    return 0;
}