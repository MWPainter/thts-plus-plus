#pragma once

#include "thts_manager.h"
#include "mo/mo_helper_templates.h"
#include "mo/mo_thts_env.h"
#include "mo/mo_thts_types.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct MoThtsManagerArgs : public ThtsManagerArgs {
        static const int reward_dim_default = -1;
        
        int reward_dim;
        MoHeuristicFnPtr mo_heuristic_fn;

        MoThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            ThtsManagerArgs(thts_env),
            reward_dim(MoThtsManagerArgs::reward_dim_default),
            mo_heuristic_fn(nullptr) {}

        virtual ~MoThtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      reward_dim:
     *          The dimension of rewards in the multi objective environment
     *      mo_heuristic_fn:
     *          A pointer to the heuristic function to use. Defaults to return a constant zero value.
     */
    class MoThtsManager : public ThtsManager {
        public:
            int reward_dim;
            MoHeuristicFnPtr mo_heuristic_fn;

            /**
             * Constructor. Initialises values directly other than random number generation.
             * 
             * If no reward dimension given, then get it from the MoThtsEnv.
             * 
             * If no heuristic function is set, then set it to the zero heuristic of the appropriate dimension. This 
             * needs to be done in the body of the constructor to allow for custom heuristics to be supplied, but also 
             * allow the default zero heuristic to match the reward_dim given
             */    
            MoThtsManager(const MoThtsManagerArgs& args) : 
                ThtsManager(args),
                reward_dim(args.reward_dim),
                mo_heuristic_fn(args.mo_heuristic_fn)
            {
                MoThtsEnv& mo_thts_env = (MoThtsEnv&) *thts_env(); 
                if (reward_dim == MoThtsManagerArgs::reward_dim_default) {
                    reward_dim = mo_thts_env.get_reward_dim();
                } else if (reward_dim != mo_thts_env.get_reward_dim()) {
                    throw std::runtime_error("Reward dim in MoThtsManager doesn't match reward dim of env.");
                }

                if (mo_heuristic_fn == nullptr) {
                    mo_heuristic_fn = get_default_mo_zero_heuristic_fn();
                }
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~MoThtsManager() = default;

        private:
            /**
             * Work around to get a default heuristic using a dynamic value (as template parameters need to be 
             * specified at compile time).
            */
            MoHeuristicFnPtr get_default_mo_zero_heuristic_fn() {
                switch (reward_dim) {
                    case 2: return helper::mo_zero_heuristic_fn<2>;
                    case 3: return helper::mo_zero_heuristic_fn<3>;
                    case 4: return helper::mo_zero_heuristic_fn<4>;
                    case 5: return helper::mo_zero_heuristic_fn<5>;
                    case 6: return helper::mo_zero_heuristic_fn<6>;
                    case 7: return helper::mo_zero_heuristic_fn<7>;
                    case 8: return helper::mo_zero_heuristic_fn<8>;
                    case 9: return helper::mo_zero_heuristic_fn<9>;
                    default: throw std::runtime_error(
                                "get_default_mo_zero_heuristic_fn doesnt contain the reward dimension you're trying to "
                                "use in include/multi_objective/mo_thts_manager, add a case to the switch block so the "
                                "compiler will generate the zero heuristic function with appropriate dimension you are "
                                "trying to use.");
                }
            }
    };
}