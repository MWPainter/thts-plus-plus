#pragma once

#include "mo/czt_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     * 
     * Make this subclass CztManager for CHMCTS, any ball list / contextual zooming args can just be ignored by an CH 
     * algorithm that doesnt use contextual zooming for trees
     */
    struct CH_MoThtsManagerArgs : public CztManagerArgs {

        CH_MoThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            CztManagerArgs(thts_env)
        {
        }

        virtual ~CH_MoThtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Make this subclass CztManager for CHMCTS, any ball list / contextual zooming args can just be ignored by an CH 
     * algorithm that doesnt use contextual zooming for trees
     * 
     * Member variables (environment):
     */
    class CH_MoThtsManager : public CztManager {
        public:

            /**
             * Constructor.
             */    
            CH_MoThtsManager(const CH_MoThtsManagerArgs& args) : 
                CztManager(args)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~CH_MoThtsManager() = default;
            
    };
} 