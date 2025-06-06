#pragma once

#include "mc_eval.h"
#include "mo/mo_thts_context.h"
#include "mo/mo_thts_manager.h"
#include "mo/mo_thts_types.h"

namespace thts {

    /**
     * MO MC Evaluator
    */
    class MoMCEvaluator : public MCEvaluator {
        protected:
            std::vector<Vec> mo_sampled_returns;
            std::vector<double> sampled_ctx_returns;
            std::vector<double> sampled_normalised_ctx_returns;
            Vec r_min;
            Vec r_max;
            bool well_spaced_eval;
            std::vector<Vec> context_weights;

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
                Vec r_max,
                bool well_spaced_eval=false);
            
            virtual ~MoMCEvaluator() = default;

            /**
             * Override of run rollouts
            */
            virtual void run_rollouts(int num_rollouts, int num_threads) override;

            /**
             * Returns the mean return of 'sampled_returns'
            */
            Vec get_mo_return_mean();
            double get_mo_return_mean(Vec context_weights);
            double get_mo_ctx_return_mean();
            double get_normalised_mo_ctx_return_mean();

            /**
                * Returns the stddev of 'sampled_returns'
            */
            Vec get_mo_return_variance();
            double get_mo_return_variance(Vec context_weights);
            double get_mo_ctx_return_variance();
            double get_normalised_mo_ctx_return_variance();
    };
}