#pragma once

#include "mo/mo_thts_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct ChThtsManagerArgs : public MoThtsManagerArgs {

        ChThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            MoThtsManagerArgs(thts_env)
        {
        }

        virtual ~ChThtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Member variables (environment):
     */
    class ChThtsManager : public MoThtsManager {
        public:

            /**
             * Constructor.
             */    
            ChThtsManager(const ChThtsManagerArgs& args) : 
                MoThtsManager(args)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ChThtsManager() = default;
            
    };
} 