#include "main/envs/entropy_trap.h"

using namespace std; 

namespace thts {
    /**
     * Construct
    */
    EntropyTrapEnv::EntropyTrapEnv(int D, int H, double final_reward) : 
        DChainEnv(D,final_reward), H(H)
    {
    }

    shared_ptr<ThtsEnv> EntropyTrapEnv::clone() {
        return make_shared<EntropyTrapEnv>(D,H,final_reward);
    }

    /**
     * -1 used for sink state when move down, D+H+1 is end of both chain
    */
    bool EntropyTrapEnv::is_sink_state(shared_ptr<const IntState> state) const {
        return state->state == -1 || state->state == D+H+1;
    }

    /**
     * Next state 
    */
    shared_ptr<const IntState> EntropyTrapEnv::sample_transition_distribution(
        shared_ptr<const IntState> state, shared_ptr<const IntAction> action) const 
    {
        if (action->action == DCHAIN_DOWN && state->state <= D) {
            return make_shared<const IntState>(-1);
        }
        return make_shared<const IntState>(state->state + 1);
    }

    /**
     * Completely redo get reward, as final reward is in a different place
    */
    double EntropyTrapEnv::get_reward(
        shared_ptr<const IntState> state, 
        shared_ptr<const IntAction> action) const 
    {
        if (action->action == DCHAIN_DOWN && state->state <= D) {
            if (state->state == D) {
                return final_reward;
            }
            return (D - state->state - 1) / (double) D;
        }
        return 0.0;
    }
}