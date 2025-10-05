#pragma once

#include "algorithms/ments/ments_manager.h"

namespace thts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct DentsManagerArgs : public MentsManagerArgs {
        static constexpr double default_init_entropy_coeff=1.0;
        static const bool use_dp_value_default=true;

        std::shared_ptr<Schedule> entropy_coeff_schedule_ptr;
        bool use_dp_value;

        DentsManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            MentsManagerArgs(thts_env),
            entropy_coeff_schedule_ptr(std::make_shared<SqrtSchedule>(default_init_entropy_coeff)),
            use_dp_value(use_dp_value_default) {}

        virtual ~DentsManagerArgs() = default;
    };

    /**
     * A specific instance of ThtsManager for DENTS algorithms.
     * 
     * The 'value_temp' refers to the temperature coefficient of entropy when computing (soft) values
     * 
     * Member variables (value temp decay):
     *      entropy_temp_decay_fn:
     *          The decay function to use for the value temp. A value of nullptr can be used to indicate that this 
     *          decay shouldn't be used (in which case this algorithm will run similarly to MENTS, but with seperately 
     *          computed value and entropy, rather than the single soft value)
     *      entropy_temp:
     *          The initial value for the value temp
     *      entropy_temp_decay_fn_min_temp:
     *          The minimum temperature to use for the value temp
     *      entropy_temp_decay_fn_x_scale:
     *          The scaling to use with the valuje_temp decay function. This is an x-axis scaling for the number of 
     *          visits input into the temp decay function
     *      value_temp_decay_root_node_visits_scale:
     *          Same as the visits scale, but for the root node. Default value of -1.0 indicates to use the same value 
     *          as 'entropy_temp_decay_fn_x_scale'
     * 
     * Member variables (values):
     *      use_dp_value:
     *          If true (default) then dynamic programming backups will be used for (*not* soft) value estimates. If 
     *          false then the average return is used. Additionally, when MentsManager::recommend_most_visited is false,
     *          this option decides if we use dp values or average returns to recommend actions.
     */
    class DentsManager : public MentsManager {
        public:
            std::shared_ptr<Schedule> entropy_coeff_schedule_ptr;
            bool use_dp_value;

            DentsManager(const DentsManagerArgs& args) :
                MentsManager(args),
                entropy_coeff_schedule_ptr(args.entropy_coeff_schedule_ptr),
                use_dp_value(args.use_dp_value) {};
    };
}