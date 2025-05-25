#pragma once

#include "thts_context.h"

#include "mo/mo_thts_manager.h"
#include "mo/mo_thts_types.h"
#include "mo/mo_helper.h"

#include <Eigen/Dense>

namespace thts {
    // forward declrs
    class MoThtsManager;
    
    /**
     * A subclass of ThtsContext that adds a weight vector for making consistent decision through a trial.
     * 
     * Member variables:
     *      context_weight: A weight to use for making contextual/consistent decisions throughout a trial
     */
    class MoThtsContext : public ThtsContext {
        public:
            Vec context_weight;

            MoThtsContext(MoThtsManager& manager);
            MoThtsContext(Vec weight);
            virtual ~MoThtsContext() = default;
    };
}