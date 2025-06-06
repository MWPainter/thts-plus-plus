#pragma once

#include "mo/algorithms/chmcts/ch_uct_manager.h"

#include "mo/mo_thts_types.h"
#include <memory>
#include <stdexcept>

namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct ChHvUctManagerArgs : public ChUctManagerArgs {

        std::shared_ptr<Vec> hv_reference_point;

        ChHvUctManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            ChUctManagerArgs(thts_env),
            hv_reference_point(nullptr)
        {
        }

        virtual ~ChHvUctManagerArgs() = default;
    };
    
    /**
     * ThtsManager for algorithms using convex hulls
     * 
     * Member variables (environment):
     */
    class ChHvUctManager : public ChUctManager {
        public:

            std::shared_ptr<Vec> hv_reference_point;

            /**
             * Constructor.
             */    
            ChHvUctManager(const ChHvUctManagerArgs& args) : 
                ChUctManager(args),
                hv_reference_point(args.hv_reference_point)
            {
                if (hv_reference_point == nullptr) {
                    throw std::runtime_error("Cannot run Hypervolume UCT without a reference point to use for hypervolume");
                }
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ChHvUctManager() = default;
            
    };
} 