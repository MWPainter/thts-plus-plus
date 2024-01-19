#pragma once

#include "mo/smt_manager.h"
#include "algorithms/common/decaying_temp.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmtBtsManagerArgs : public SmtManagerArgs {
        static constexpr double temp_default=1.0;
        static constexpr double epsilon_default=0.5;

        static constexpr TempDecayFnPtr temp_decay_fn_default=nullptr;
        static constexpr double temp_decay_min_temp_default=1.0e-6;
        static constexpr double temp_decay_visits_scale_default=1.0;
        static constexpr double temp_decay_root_node_visits_scale_default=-1.0;

        double temp; 
        double epsilon;

        TempDecayFnPtr temp_decay_fn;
        double temp_decay_min_temp;
        double temp_decay_visits_scale;
        double temp_decay_root_node_visits_scale;

        SmtBtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            SmtThtsManagerArgs(thts_env, default_q_value),
            temp(temp_default),
            epsilon(epsilon_default),
            temp_decay_fn(temp_decay_fn_default),
            temp_decay_min_temp(temp_decay_min_temp_default),
            temp_decay_visits_scale(temp_decay_visits_scale_default),
            temp_decay_root_node_visits_scale(temp_decay_root_node_visits_scale_default)
        {
        }

        virtual ~SmtBtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      bias:
     *          The CZT bias
     */
    class SmtBtsManager : public SmtManager {
        public:
            double temp; 
            double epsilon;

            TempDecayFnPtr temp_decay_fn;
            double temp_decay_min_temp;
            double temp_decay_visits_scale;
            double temp_decay_root_node_visits_scale;

            /**
             * Constructor.
             */    
            SmtBtsManager(const SmtBtsManagerArgs& args) : 
                SmtThtsManager(args),
                temp(args.temp),
                epsilon(args.epsilon),
                temp_decay_fn(args.temp_decay_fn),
                temp_decay_min_temp(args.temp_decay_min_temp),
                temp_decay_visits_scale(args.temp_decay_visits_scale),
                temp_decay_root_node_visits_scale(args.temp_decay_root_node_visits_scale)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmtBtsManager() = default;
            
    };
}