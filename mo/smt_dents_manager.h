#pragma once

#include "mo/smt_bts_manager.h"
#include "algorithms/common/decaying_temp.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmtDentsManagerArgs : public SmtBtsManagerArgs {
        static constexpr TempDecayFnPtr value_temp_decay_fn_default=decayed_temp_inv_sqrt;
        static constexpr double value_temp_init_default=1.0;
        static constexpr double value_temp_decay_min_temp_default=1.0e-6;
        static constexpr double value_temp_decay_visits_scale_default=1.0;
        static constexpr double value_temp_decay_root_node_visits_scale_default=-1.0;
        static const bool use_dp_value_default=true;

        TempDecayFnPtr value_temp_decay_fn;
        double value_temp_init;
        double value_temp_decay_min_temp;
        double value_temp_decay_visits_scale;

        SmtDentsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            SmtBtsManagerArgs(thts_env, default_q_value),
            value_temp_decay_fn(value_temp_decay_fn_default),
            value_temp_init(value_temp_init_default),
            value_temp_decay_min_temp(value_temp_decay_min_temp_default),
            value_temp_decay_visits_scale(value_temp_decay_visits_scale_default)
        {
        }

        virtual ~SmtDentsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      bias:
     *          The CZT bias
     */
    class SmtDentsManager : public SmtBtsManager {
        public:
            TempDecayFnPtr value_temp_decay_fn;
            double value_temp_init;
            double value_temp_decay_min_temp;
            double value_temp_decay_visits_scale;

            /**
             * Constructor.
             */    
            SmtDentsManager(const SmtDentsManagerArgs& args) : 
                SmtBtsManager(args),
                value_temp_decay_fn(args.value_temp_decay_fn),
                value_temp_init(args.value_temp_init),
                value_temp_decay_min_temp(args.value_temp_decay_min_temp),
                value_temp_decay_visits_scale(args.value_temp_decay_visits_scale)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmtDentsManager() = default;
            
    };
}