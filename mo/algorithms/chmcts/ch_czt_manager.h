#pragma once

#include "mo/algorithms/contextual_zooming/czt_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct ChCztManagerArgs : public CztManagerArgs {

        ChCztManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            CztManagerArgs(thts_env)
        {
        }

        virtual ~ChCztManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Member variables (environment):
     */
    class ChCztManager : public CztManager {
        public:

            /**
             * Constructor.
             */    
            ChCztManager(const ChCztManagerArgs& args) : 
                CztManager(args)
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
            virtual ~ChCztManager() = default;
            
    };
} 