#include "thts_env.h"

#include "helper_templates.h"

#include <stdexcept>

using namespace std;

namespace thts {
    /**
     * Constructor
     */
    MoThtsEnv::MoThtsEnv(bool is_fully_observable) : ThtsEnv(is_fully_observable) 
    {
    }

    /**
     * Make get_reward_itfc raise an exception
    */
    double MoThtsEnv::get_reward_itfc(
        std::shared_ptr<const State> state, 
        std::shared_ptr<const Action> action, 
        std::shared_ptr<const Observation> observation=nullptr) const 
    {
        throw runtime_error("Shouldn't call get_reward_itfc from a multi-objective env.");
    }
} 