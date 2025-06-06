#pragma once

#include "mo/algorithms/chmcts/ch_thts_manager.h"

#include "algorithms/common/decaying_temp.h"

namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct ChBtsManagerArgs : public ChThtsManagerArgs {
        static const bool normalise_q_values_default=true;
        static constexpr double temp_default=1.0;
        static constexpr double epsilon_default=0.5;
        static constexpr double max_explore_prob_default=1.0;

        static constexpr TempDecayFnPtr temp_decay_fn_default=nullptr;
        static constexpr double temp_decay_fn_min_temp_default=1.0e-6;
        static constexpr double temp_decay_fn_x_scale_default=1.0;

        static constexpr double default_q_value_default=0.0;

        bool normalise_q_values;
        double temp;
        double epsilon;
        double max_explore_prob;

        TempDecayFnPtr temp_decay_fn;
        double temp_decay_fn_min_temp;
        double temp_decay_fn_x_scale;

        double default_q_value;

        ChBtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            ChThtsManagerArgs(thts_env),
            normalise_q_values(normalise_q_values_default),
            temp(temp_default),
            epsilon(epsilon_default),
            max_explore_prob(max_explore_prob_default),
            temp_decay_fn(temp_decay_fn_default),
            temp_decay_fn_min_temp(temp_decay_fn_min_temp_default),
            temp_decay_fn_x_scale(temp_decay_fn_x_scale_default),
            default_q_value(default_q_value_default)
        {
        }

        virtual ~ChBtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Member variables (environment):
     */
    class ChBtsManager : public ChThtsManager {
        public:
            bool normalise_q_values;
            double temp;
            double epsilon;
            double max_explore_prob;

            TempDecayFnPtr temp_decay_fn;
            double temp_decay_fn_min_temp;
            double temp_decay_fn_x_scale;

            double default_q_value;

            /**
             * Constructor.
             */    
            ChBtsManager(const ChBtsManagerArgs& args) : 
                ChThtsManager(args),
                normalise_q_values(args.normalise_q_values),
                temp(args.temp),
                epsilon(args.epsilon),
                max_explore_prob(args.max_explore_prob),
                temp_decay_fn(args.temp_decay_fn),
                temp_decay_fn_min_temp(args.temp_decay_fn_min_temp),
                temp_decay_fn_x_scale(args.temp_decay_fn_x_scale),
                default_q_value(args.default_q_value)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ChBtsManager() = default;
            
    };
} 