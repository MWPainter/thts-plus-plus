#pragma once

#include "mo/mo_thts_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct CH_MoThtsManagerArgs : public MoThtsManagerArgs {

        CH_MoThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            MoThtsManagerArgs(thts_env)
        {
        }

        virtual ~CH_MoThtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Member variables (environment):
     */
    class CH_MoThtsManager : public MoThtsManager {
        public:

            /**
             * Constructor.
             */    
            CH_MoThtsManager(const CH_MoThtsManagerArgs& args) : 
                MoThtsManager(args)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~CH_MoThtsManager() = default;
            
    };
} 