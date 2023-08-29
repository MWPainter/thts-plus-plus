#include "multi_objective/mo_helper_templates.h"

using namespace std;

namespace thts::helper {
    /**
     * Implementation of the default zero heuristic function.
     */
    template <unsigned int dim>
    Eigen::VectorXd mo_zero_heuristic_fn(shared_ptr<const State> state, shared_ptr<ThtsEnv> env) {
        return Eigen::VectorXd::Constant(dim, 0.0);
    }
}