#pragma once

#include "algorithms/uct/uct_manager.h"

#include <stdexcept>

namespace thts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct HmctsManagerArgs : public UctManagerArgs {
        static const int total_budget_default = -1;
        static const int uct_budget_threshold_default = 100;

        int total_budget;
        int uct_budget_threshold;

        HmctsManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            UctManagerArgs(thts_env),
            total_budget(total_budget_default),
            uct_budget_threshold(uct_budget_threshold_default) {}

        virtual ~HmctsManagerArgs() = default;
    };

    /**
     * A specific instance of ThtsManager for HMCTS algorithms.
     * 
     * Member variables:
     *      total_budget:
     *          The total budget for running HMCTS, i.e. the total number of trials we will run
     *      uct_budget_threshold:
     *          The threshold that decides when to switch to using PUCT from SHOT
     */
    class HmctsManager : public UctManager {
        public:
            int total_budget;
            int uct_budget_threshold;

            HmctsManager(const HmctsManagerArgs& args) :
                UctManager(args),
                total_budget(args.total_budget),
                uct_budget_threshold(args.uct_budget_threshold) 
            {
                // Required args
                if (args.total_budget == HmctsManagerArgs::total_budget_default) {
                    throw std::invalid_argument("Total budget is required arg for HMCTS");
                }
            };
    };
}