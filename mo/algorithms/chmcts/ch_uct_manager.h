#pragma once

#include "mo/algorithms/chmcts/ch_thts_manager.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct ChUctManagerArgs : public ChThtsManagerArgs {
        static constexpr double ADAPTIVE_BIAS_MIN_BIAS = 0.001;

        static const bool adaptive_bias_default=false;
        static const bool normalize_Q_values_in_selection_default=true;
        static constexpr double bias_default=4.0;

        bool adaptive_bias;
        bool normalize_Q_values_in_selection;
        double bias;

        ChUctManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            ChThtsManagerArgs(thts_env),
            adaptive_bias(adaptive_bias_default),
            normalize_Q_values_in_selection(normalize_Q_values_in_selection_default),
            bias(bias_default)
        {
        }

        virtual ~ChUctManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Member variables (environment):
     */
    class ChUctManager : public ChThtsManager {
        public:
            static constexpr double ADAPTIVE_BIAS_MIN_BIAS = ChUctManagerArgs::ADAPTIVE_BIAS_MIN_BIAS;

            bool normalize_Q_values_in_selection;
            bool adaptive_bias;
            double bias;

            /**
             * Constructor.
             */    
            ChUctManager(const ChUctManagerArgs& args) : 
                ChThtsManager(args),
                normalize_Q_values_in_selection(args.normalize_Q_values_in_selection),
                adaptive_bias(args.adaptive_bias),
                bias(args.bias)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ChUctManager() = default;
            
    };
} 