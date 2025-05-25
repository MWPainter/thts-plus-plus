#include "mo/mo_thts_context.h"

#include <iostream>

using namespace std;

namespace thts {

    MoThtsContext::MoThtsContext(MoThtsManager& manager) : 
        ThtsContext(), 
        context_weight(thts::helper::sample_uniform_random_simplex_vector(manager,manager.reward_dim)) 
    {
    } 

    MoThtsContext::MoThtsContext(Vec weight) : 
        ThtsContext(), 
        context_weight(weight) 
    {
    } 
}
