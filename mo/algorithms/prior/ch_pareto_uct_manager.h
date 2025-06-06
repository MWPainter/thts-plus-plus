#pragma once

#include "mo/algorithms/chmcts/ch_uct_manager.h"

#include "mo/mo_thts_types.h"
#include <memory>
#include <stdexcept>

namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct ChParetoUctManagerArgs : public ChUctManagerArgs {

        ChParetoUctManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            ChUctManagerArgs(thts_env)
        {
        }

        virtual ~ChParetoUctManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Member variables (environment):
     */
    class ChParetoUctManager : public ChUctManager {
        public:

            /**
             * Constructor.
             */    
            ChParetoUctManager(const ChParetoUctManagerArgs& args) : 
                ChUctManager(args)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ChParetoUctManager() = default;
            
    };
} 