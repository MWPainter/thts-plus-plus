#include "mo/mo_thts_manager.h"

#include <sstream>

using namespace std;

namespace thts {


    /**
     * Constructor. Initialises values directly other than random number generation.
     */    
    MoThtsManager::MoThtsManager(const MoThtsManagerArgs& args) :
        ThtsManager(args),
        reward_dim(args.reward_dim),
        mo_heuristic_fn(args.mo_heuristic_fn)
    {
        MoThtsEnv& mo_thts_env = *dynamic_pointer_cast<MoThtsEnv>(thts_env()); 
        if (reward_dim == MoThtsManagerArgs::reward_dim_default) {
            reward_dim = mo_thts_env.get_reward_dim();
        } else if (reward_dim != mo_thts_env.get_reward_dim()) {
            throw std::runtime_error("Reward dim in MoThtsManager doesn't match reward dim of env.");
        }

        if (mo_heuristic_fn == nullptr) {
            mo_heuristic_fn = get_default_mo_zero_heuristic_fn();
        }
    }

    MoHeuristicFnPtr MoThtsManager::get_default_mo_zero_heuristic_fn() {
        switch (reward_dim) {
            case 2: 
                return helper::mo_zero_heuristic_fn<2>;
            case 3: 
                return helper::mo_zero_heuristic_fn<3>;
            case 4: 
                return helper::mo_zero_heuristic_fn<4>;
            case 5: 
                return helper::mo_zero_heuristic_fn<5>;
            case 6: 
                return helper::mo_zero_heuristic_fn<6>;
            case 7: 
                return helper::mo_zero_heuristic_fn<7>;
            case 8: 
                return helper::mo_zero_heuristic_fn<8>;
            case 9: 
                return helper::mo_zero_heuristic_fn<9>;
            default: 
                stringstream ss;
                ss << "get_default_mo_zero_heuristic_fn doesnt contain the reward dimension (" << reward_dim << ") "
                    << "you're trying to use in include/multi_objective/mo_thts_manager, add a case to the switch "
                    << "block so the compiler will generate the zero heuristic function with appropriate dimension you "
                    << "are trying to use.";
                throw runtime_error(ss.str());
        }
    }
}