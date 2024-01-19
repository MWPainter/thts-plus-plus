#pragma once

#include "mc_eval.h"
#include "mo/mo_thts_context.h"
#include "mo/mo_thts_manager.h"

#include <Eigen/Dense>

namespace thts {

    /**
     * MO MC Evaluator
    */
    class MoMCEvaluator : virtual public MCEvaluator {
        protected:
            std::vector<Eigen::ArrayXd> mo_sampled_returns;
            std::vector<double> sampled_ctx_returns;
            std::vector<double> sampled_normalised_ctx_returns;
            Eigen::ArrayXd r_min;
            Eigen::ArrayXd r_max;

            /**
             * Runs a single rollout and stores the result in 'sampled_returns'.
            */
            virtual void run_rollout(int thread_id, EvalPolicy& thread_policy) override;



        public:
            MoMCEvaluator(
                std::shared_ptr<EvalPolicy> eval_policy,
                int max_trial_length,
                std::shared_ptr<MoThtsManager> manager,
                Eigen::ArrayXd r_min,
                Eigen::ArrayXd r_max);
            
            virtual ~MoMCEvaluator() = default;

            /**
             * Returns the mean return of 'sampled_returns'
            */
            Eigen::ArrayXd get_mean_mo_return();
            double get_mean_mo_return(Eigen::ArrayXd context_weights);
            double get_mean_mo_ctx_return();
            double get_mean_mo_normalised_ctx_return();

            /**
                * Returns the stddev of 'sampled_returns'
            */
            Eigen::ArrayXd get_stddev_mo_return();
            double get_stddev_mo_return(Eigen::ArrayXd context_weights);
            double get_stddev_mean_mo_ctx_return();
            double get_stddev_mean_mo_normalised_ctx_return();
    };
}