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
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      num_backups_before_allowed_to_split:
     *          The number of backups that have to be performed at a CZ_Ball before it is allowed to 'split' and 
     *          create child balls.
     */
    class CH_MoThtsManager : public MoThtsManager {
        public:
            int num_backups_before_allowed_to_split;

            /**
             * Constructor.
             */    
            CH_MoThtsManager(const CH_MoThtsManagerArgs& args) : 
                MoThtsManager(args),
                num_backups_before_allowed_to_split(args.num_backups_before_allowed_to_split)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~CH_MoThtsManager() = default;
            
    };
}