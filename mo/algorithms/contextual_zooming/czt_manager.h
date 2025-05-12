#pragma once

#include "mo/algorithms/contextual_zooming/bl_thts_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct CztManagerArgs : public BlThtsManagerArgs {
        static constexpr double bias_default=4.0;

        double bias;

        CztManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            BlThtsManagerArgs(thts_env),
            bias(CztManagerArgs::bias_default)
        {
        }

        virtual ~CztManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      bias:
     *          The CZT bias
     */
    class CztManager : public BlThtsManager {
        public:
            double bias;

            /**
             * Constructor.
             */    
            CztManager(const CztManagerArgs& args) : 
                BlThtsManager(args),
                bias(args.bias)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~CztManager() = default;
            
    };
}