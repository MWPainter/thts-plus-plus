#pragma once

#include "mo/algorithms/simplex_maps/sm_bts_manager.h"
#include "algorithms/common/decaying_temp.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmDentsManagerArgs : public SmBtsManagerArgs {
        static constexpr TempDecayFnPtr entropy_temp_decay_fn_default=decayed_temp_inv_sqrt;
        static constexpr double entropy_temp_default=1.0;
        static constexpr double entropy_temp_decay_fn_min_temp_default=1.0e-6;
        static constexpr double entropy_temp_decay_fn_x_scale_default=1.0;
        static constexpr double value_temp_decay_root_node_visits_scale_default=-1.0;
        static const bool use_dp_value_default=true;

        TempDecayFnPtr entropy_temp_decay_fn;
        double entropy_temp;
        double entropy_temp_decay_fn_min_temp;
        double entropy_temp_decay_fn_x_scale;

        SmDentsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            SmBtsManagerArgs(thts_env, default_q_value),
            entropy_temp_decay_fn(entropy_temp_decay_fn_default),
            entropy_temp(entropy_temp_default),
            entropy_temp_decay_fn_min_temp(entropy_temp_decay_fn_min_temp_default),
            entropy_temp_decay_fn_x_scale(entropy_temp_decay_fn_x_scale_default)
        {
        }

        virtual ~SmDentsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      bias:
     *          The CZT bias
     */
    class SmDentsManager : public SmBtsManager {
        public:
            TempDecayFnPtr entropy_temp_decay_fn;
            double entropy_temp;
            double entropy_temp_decay_fn_min_temp;
            double entropy_temp_decay_fn_x_scale;

            /**
             * Constructor.
             */    
            SmDentsManager(const SmDentsManagerArgs& args) : 
                SmBtsManager(args),
                entropy_temp_decay_fn(args.entropy_temp_decay_fn),
                entropy_temp(args.entropy_temp),
                entropy_temp_decay_fn_min_temp(args.entropy_temp_decay_fn_min_temp),
                entropy_temp_decay_fn_x_scale(args.entropy_temp_decay_fn_x_scale)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmDentsManager() = default;
            
    };
}