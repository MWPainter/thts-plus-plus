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
            ThtsManagerArgs(std::static_pointer_cast<ThtsEnv>(thts_env)),
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
            MoThtsManager(const MoThtsManagerArgs& args);

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~MoThtsManager() = default;

        private:
            /**
             * Work around to get a default heuristic using a dynamic value (as template parameters need to be 
             * specified at compile time).
            */
            MoHeuristicFnPtr get_default_mo_zero_heuristic_fn();
    };
}