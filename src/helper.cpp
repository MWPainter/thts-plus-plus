#include "helper.h"

using namespace std;

namespace thts::helper {
    /**
     * Implementation of the default zero heuristic function.
     */
    double zero_heuristic_fn(shared_ptr<const State> state) {
        return 0.0;
    }
}