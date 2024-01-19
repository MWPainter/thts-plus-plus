#pragma once

#include "mo/mo_thts_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmtThtsManagerArgs : public MoThtsManagerArgs {
        static const int num_backups_before_allowed_to_split_default = -1;

        int num_backups_before_allowed_to_split;
        Eigen::ArrayXd default_q_value;

        SmtThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            MoThtsManagerArgs(thts_env),
            num_backups_before_allowed_to_split(num_backups_before_allowed_to_split_default), 
            default_q_value(default_q_value)
        {
        }

        virtual ~SmtThtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      num_backups_before_allowed_to_split:
     *          The number of backups that have to be performed at a CZ_Ball before it is allowed to 'split' and 
     *          create child balls.
     */
    class SmtThtsManager : public MoThtsManager {
        public:
            int num_backups_before_allowed_to_split;
            Eigen::ArrayXd default_q_value;

            /**
             * Constructor.
             */    
            SmtThtsManager(const SmtThtsManagerArgs& args) : 
                MoThtsManager(args),
                num_backups_before_allowed_to_split(args.num_backups_before_allowed_to_split),
                default_q_value(args.default_q_value)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmtThtsManager() = default;
            
    };
}