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
    struct ChChebyUctManagerArgs : public ChUctManagerArgs {

        static constexpr double cheby_delta_default = 0.01;
        static const bool use_standard_cheby_scalarization_default = false;


        double cheby_delta;
        bool use_standard_cheby_scalarization;
        std::shared_ptr<Vec> standard_cheby_reference_point;

        ChChebyUctManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            ChUctManagerArgs(thts_env),
            cheby_delta(cheby_delta_default),
            use_standard_cheby_scalarization(use_standard_cheby_scalarization_default),
            standard_cheby_reference_point(nullptr)
        {
        }

        virtual ~ChChebyUctManagerArgs() = default;
    };
    
    /**
     *
     */
    class ChChebyUctManager : public ChUctManager {
        public:

            double cheby_delta;
            bool use_standard_cheby_scalarization;
            std::shared_ptr<Vec> standard_cheby_reference_point;

            /**
             * Constructor.
             */    
            ChChebyUctManager(const ChChebyUctManagerArgs& args) : 
                ChUctManager(args),
                cheby_delta(args.cheby_delta),
                use_standard_cheby_scalarization(args.use_standard_cheby_scalarization),
                standard_cheby_reference_point(args.standard_cheby_reference_point)
            {
                if (use_standard_cheby_scalarization && args.standard_cheby_reference_point == nullptr) {
                    throw std::runtime_error("Cannot run Standard Cheby UCT without a reference point to use for standard cheby scalarization");
                }
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ChChebyUctManager() = default;
            
    };
} 