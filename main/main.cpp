#include "main/run_id.h"
#include "main/run_expr.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        throw runtime_error("Expecting exactly one argument specifying valid expr id");
    }

    shared_ptr<vector<RunID>> run_ids = thts::get_run_ids_from_expr_id(argv[1]);
    thts::run_exprs(run_ids);

    return 0;
}