#pragma once

#include "mo/czt_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct ChmctsManagerArgs : public CztManagerArgs {

        ChmctsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            CztManagerArgs(thts_env)
        {
        }

        virtual ~ChmctsManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Member variables (environment):
     */
    class ChmctsManager : public CztManager {
        public:

            /**
             * Constructor.
             */    
            ChmctsManager(const ChmctsManagerArgs& args) : 
                CztManager(args)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ChmctsManager() = default;
            
    };
} 