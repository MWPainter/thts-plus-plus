#pragma once

#include "mo/mo_thts_manager.h"
#include "mo/algorithms/chmcts/ch_thts_manager.h"




/**
 * Hacky thing:
 * - so that ChCzt can be implmented where ChCzt nodes have member Czt nodes, make this a subclass of ChThtsManager
 * - this way we can easily cast to ChThtsManager and CztManager in ChCzt
 * - longer term support should do something more sensible, possibly with some code bloat but cleaner
 */


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct BlThtsManagerArgs : public ChThtsManagerArgs {
        static const int num_backups_before_allowed_to_split_default = -1;

        int num_backups_before_allowed_to_split;

        BlThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            ChThtsManagerArgs(thts_env),
            num_backups_before_allowed_to_split(BlThtsManagerArgs::num_backups_before_allowed_to_split_default) 
        {
        }

        virtual ~BlThtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      num_backups_before_allowed_to_split:
     *          The number of backups that have to be performed at a CzBall before it is allowed to 'split' and 
     *          create child balls.
     */
    class BlThtsManager : public ChThtsManager {
        public:
            int num_backups_before_allowed_to_split;

            /**
             * Constructor.
             */    
            BlThtsManager(const BlThtsManagerArgs& args) : 
                ChThtsManager(args),
                num_backups_before_allowed_to_split(args.num_backups_before_allowed_to_split)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~BlThtsManager() = default;
            
    };
}