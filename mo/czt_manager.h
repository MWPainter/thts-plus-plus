#pragma once

#include "mo/bl_thts_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct CztManagerArgs : public BL_MoThtsManagerArgs {
        static constexpr double bias_default=4.0;

        double bias;

        CztManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            BL_MoThtsManagerArgs(thts_env),
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
    class CztManager : public BL_MoThtsManager {
        public:
            double bias;

            /**
             * Constructor.
             */    
            CztManager(const CztManagerArgs& args) : 
                BL_MoThtsManager(args),
                bias(args.bias)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~CztManager() = default;
            
    };
}