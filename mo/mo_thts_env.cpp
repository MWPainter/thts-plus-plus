#include "mo/mo_thts_env.h"

#include <stdexcept>

using namespace std;


namespace thts {
    /**
     * Constructor
     */
    MoThtsEnv::MoThtsEnv(int reward_dim, bool _is_fully_observable) : 
        ThtsEnv(_is_fully_observable),
        reward_dim(reward_dim)
    {
    }

    MoThtsEnv::MoThtsEnv(MoThtsEnv& other) :
        ThtsEnv(other._is_fully_observable),
        reward_dim(other.reward_dim)
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
        ThtsEnvContext& ctx) const 
    {
        throw runtime_error("Shouldn't call get_reward_itfc from a multi-objective env.");
    }
} 