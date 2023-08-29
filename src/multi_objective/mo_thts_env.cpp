#include "multi_objective/mo_thts_env.h"

#include <stdexcept>

using namespace std;

namespace thts {
    /**
     * Constructor
     */
    MoThtsEnv::MoThtsEnv(int reward_dim, bool is_fully_observable) : 
        ThtsEnv(is_fully_observable),
        reward_dim(reward_dim)
    {
    }

    /**
     * Get reward dim
    */
    int MoThtsEnv::get_reward_dim() {
        return reward_dim;
    }

    /**
     * Make get_reward_itfc raise an exception
    */
    double MoThtsEnv::get_reward_itfc(
        std::shared_ptr<const State> state, 
        std::shared_ptr<const Action> action, 
        std::shared_ptr<const Observation> observation) const 
    {
        throw runtime_error("Shouldn't call get_reward_itfc from a multi-objective env.");
    }
} 