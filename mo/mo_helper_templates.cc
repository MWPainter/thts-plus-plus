#include "mo/mo_helper_templates.h"


namespace thts::helper {
    using namespace std;

    /**
     * Implementation of the default zero heuristic function.
     */
    template <unsigned int dim>
    Eigen::ArrayXd mo_zero_heuristic_fn(shared_ptr<const State> state, shared_ptr<ThtsEnv> env) {
        return Eigen::ArrayXd::Zero(dim);
    }
}