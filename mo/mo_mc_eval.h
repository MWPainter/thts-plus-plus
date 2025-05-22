#pragma once

#include "mc_eval.h"
#include "mo/mo_thts_context.h"
#include "mo/mo_thts_manager.h"
#include "mo/mo_thts_types.h"

namespace thts {

    /**
     * MO MC Evaluator
    */
    class MoMCEvaluator : virtual public MCEvaluator {
        protected:
            std::vector<Vec> mo_sampled_returns;
            std::vector<double> sampled_ctx_returns;
            std::vector<double> sampled_normalised_ctx_returns;
            Vec r_min;
            Vec r_max;

            /**
             * Runs a single rollout and stores the result in 'sampled_returns'.
            */
            virtual void run_rollout(int thread_id, EvalPolicy& thread_policy) override;



        public:
            MoMCEvaluator(
                std::shared_ptr<EvalPolicy> eval_policy,
                int max_trial_length,
                std::shared_ptr<MoThtsManager> manager,
                Vec r_min,
                Vec r_max);
            
            virtual ~MoMCEvaluator() = default;

            /**
             * Returns the mean return of 'sampled_returns'
            */
            Vec get_mean_mo_return();
            double get_mean_mo_return(Vec context_weights);
            double get_mean_mo_ctx_return();
            double get_mean_mo_normalised_ctx_return();

            /**
                * Returns the stddev of 'sampled_returns'
            */
            Vec get_stddev_mo_return();
            double get_stddev_mo_return(Vec context_weights);
            double get_stddev_mean_mo_ctx_return();
            double get_stddev_mean_mo_normalised_ctx_return();
    };
}