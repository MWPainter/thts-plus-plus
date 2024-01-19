#pragma once

#include "mo/mo_thts_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct BL_MoThtsManagerArgs : public MoThtsManagerArgs {
        static const int num_backups_before_allowed_to_split_default = -1;

        int num_backups_before_allowed_to_split;

        BL_MoThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            MoThtsManagerArgs(thts_env),
            num_backups_before_allowed_to_split(BL_MoThtsManagerArgs::num_backups_before_allowed_to_split_default) 
        {
        }

        virtual ~BL_MoThtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      num_backups_before_allowed_to_split:
     *          The number of backups that have to be performed at a CZ_Ball before it is allowed to 'split' and 
     *          create child balls.
     */
    class BL_MoThtsManager : public MoThtsManager {
        public:
            int num_backups_before_allowed_to_split;

            /**
             * Constructor.
             */    
            BL_MoThtsManager(const BL_MoThtsManagerArgs& args) : 
                MoThtsManager(args),
                num_backups_before_allowed_to_split(args.num_backups_before_allowed_to_split)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~BL_MoThtsManager() = default;
            
    };
}