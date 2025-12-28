#pragma once

#include "mo/algorithms/contextual_zooming/bl_thts_manager.h"

#include <stdexcept>


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct CztManagerArgs : public BlThtsManagerArgs {
        static const bool use_doubling_N_term_default=false;
        static const int min_log2_N_default=0;
        static constexpr double bias_default=4.0;

        bool use_doubling_N_term;
        int min_log2_N;
        double bias;

        CztManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            BlThtsManagerArgs(thts_env),
            use_doubling_N_term(CztManagerArgs::use_doubling_N_term_default),
            min_log2_N(CztManagerArgs::min_log2_N_default),
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
    class CztManager : public BlThtsManager {
        public:
            bool use_doubling_N_term;
            int min_log2_N;
            double bias;

            /**
             * Constructor.
             */    
            CztManager(const CztManagerArgs& args) : 
                BlThtsManager(args),
                use_doubling_N_term(args.use_doubling_N_term),
                min_log2_N(args.min_log2_N),
                bias(args.bias)
            {
                if (use_doubling_N_term && min_log2_N < 0) 
                {
                    throw std::invalid_argument("min_log2_N must be non-negative if use_doubling_N_term is true");
                }
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~CztManager() = default;
            
    };
}