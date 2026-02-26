#pragma once

#include "mo/algorithms/simplex_maps/sm_bts_manager.h"
#include "algorithms/common/decaying_temp.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmDentsManagerArgs : public SmBtsManagerArgs {
        static constexpr double default_init_entropy_coeff=1.0;
        static constexpr double default_entropy_coeff_decay_rate=1.0;
        static const bool normalise_entropy_before_adding_default=true;

        std::shared_ptr<Schedule> entropy_coeff_schedule_ptr;
        bool normalise_entropy_before_adding;

        SmDentsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            SmBtsManagerArgs(thts_env, default_q_value),
            entropy_coeff_schedule_ptr(std::make_shared<SqrtSchedule>(default_init_entropy_coeff,default_entropy_coeff_decay_rate)),
            normalise_entropy_before_adding(normalise_entropy_before_adding_default)
        {
        }

        virtual ~SmDentsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      bias:
     *          The CZT bias
     */
    class SmDentsManager : public SmBtsManager {
        public:
            std::shared_ptr<Schedule> entropy_coeff_schedule_ptr;
            bool normalise_entropy_before_adding;

            /**
             * Constructor.
             */    
            SmDentsManager(const SmDentsManagerArgs& args) : 
                SmBtsManager(args),
                entropy_coeff_schedule_ptr(args.entropy_coeff_schedule_ptr),
                normalise_entropy_before_adding(args.normalise_entropy_before_adding)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmDentsManager() = default;
            
    };
}