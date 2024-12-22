#pragma once

#include "mo/mo_thts_manager.h"

#include <Eigen/Dense>

namespace thts {
    // forward declrs
    class MoThtsManager;
    
    /**
     * A subclass of ThtsEnvContext that adds a weight vector for making consistent decision through a trial.
     * 
     * Member variables:
     *      context_weight: A weight to use for making contextual/consistent decisions throughout a trial
     */
    class MoThtsContext : public ThtsEnvContext {
        public:
            Eigen::ArrayXd context_weight;

            MoThtsContext(MoThtsManager& manager);
            MoThtsContext(Eigen::ArrayXd weight);
            virtual ~MoThtsContext() = default;
        
            static Eigen::ArrayXd sample_uniform_random_simplex_for_weight(MoThtsManager& manager);
    };
}