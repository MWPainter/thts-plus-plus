#pragma once

#include "algorithms/uct/uct_manager.h"

namespace thts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct PuctManagerArgs : public UctManagerArgs {
        static constexpr double puct_power_default=0.5;

        double puct_power;

        PuctManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            UctManagerArgs(thts_env),
            puct_power(puct_power_default) {}

        virtual ~PuctManagerArgs() = default;
    };

    /**
     * A specific instance of ThtsManager for PUCT algorithms.
     * 
     * Member variables:
     *      puct_power:
     *          The power to use in the polynomial term of PUCT
     */
    class PuctManager : public UctManager {
        public:
            double puct_power;

            PuctManager(const PuctManagerArgs& args) :
                UctManager(args),
                puct_power(args.puct_power) {};
    };
}