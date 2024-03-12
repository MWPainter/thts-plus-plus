#pragma once

#include "mo/czt_manager.h"
#include "mo/ch_thts_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct ChmctsManagerArgs : public CH_MoThtsManagerArgs {

        ChmctsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            CH_MoThtsManagerArgs(thts_env)
        {
        }

        virtual ~ChmctsManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Member variables (environment):
     */
    class ChmctsManager : public CH_MoThtsManager {
        public:

            /**
             * Constructor.
             */    
            ChmctsManager(const ChmctsManagerArgs& args) : 
                CH_MoThtsManager(args)
            {
                if (args.use_transposition_table) {
                    throw std::runtime_error("CHMCTS isnt implemented in a way that is compatible with transposition "
                        "tables because the transposition table will try to store Czt and Chmcts nodes that will "
                        "overwrite each other in the table.");
                }
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ChmctsManager() = default;
            
    };
} 