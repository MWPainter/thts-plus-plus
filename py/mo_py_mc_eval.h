#pragma once

#include "mo/mo_mc_eval.h"
#include "py/py_mc_eval.h"

namespace thts::python {
    using namespace thts;

    /**
     * MO Py MC Evaluator
    */
    class MoPyMCEvaluator : public MoMCEvaluator, public PyMCEvaluator {
        protected:
            std::vector<Eigen::ArrayXd> mo_sampled_returns;
            std::vector<double> sampled_ctx_returns;
            std::vector<double> sampled_normalised_ctx_returns;
            Eigen::ArrayXd r_min;
            Eigen::ArrayXd r_max;

            /**
             * Point to MO override
            */
            virtual void run_rollout(int thread_id, EvalPolicy& thread_policy) override;
            /**
             * Point to Py override
            */
            virtual void thread_run_rollouts(
                int total_rollouts, 
                int thread_id, 
                int num_threads, 
                std::shared_ptr<EvalPolicy> thread_policy) override;
        public:
            virtual void run_rollouts(int num_rollouts, int num_threads) override;



        public:
            MoPyMCEvaluator(
                std::shared_ptr<EvalPolicy> eval_policy,
                int max_trial_length,
                std::shared_ptr<MoThtsManager> manager,
                Eigen::ArrayXd r_min,
                Eigen::ArrayXd r_max);
            
            virtual ~MoPyMCEvaluator() = default;
    };
}