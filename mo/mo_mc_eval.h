#pragma once

#include "mc_eval.h"
#include "mo/mo_thts_context.h"
#include "mo/mo_thts_manager.h"
#include "mo/mo_thts_types.h"

namespace thts {

    /**
     * MO MC Evaluator

     If normalized_returns is true, then the evaluation is run in a normalised space
     Normalised space will translate by - r_min and scale by 1 / (r_max - r_min)

     Prereq linear algebra:
     If a space is scaled my matrix S, so that x' = Sx
     Have line w^T x = c and want to find line ater that scaling is applied
     Then x = S^-1 x' in w^T x = c gives w^T S^-1 x' = c
     Which is also (S^-T w)^T x' = c

     If normalised_value_space is true, then the following proceedure is used:
     - sample context weights from simplex (for normalised returns)
     - algorithms use a scale of (r_max - r_min)
     - so algorithm is passed normalised(w / (r_max - r_min)) as the context weight
     - the algorithm is run and gives a return R
     - return is normalised to be between 0 and 1 by R_norm = (R - r_min) / (r_max - r_min)
     - the contextual return is then computed as w^T R_norm

     If normalised returns is false, then we sample context weights from simplex as normal
     Algorithms are given the original context weights and return an unnormalised return R
     And then compute the contextual return as w^T R

     We also compute in this case a reweighted contextual return, where the the computation is done in the normalised space
     NOTE in this case the context weights are not uniformly sampled over the simplex
     Proceedure is as follows:
     - sample context weights from simplex
     - algorithms are given the original context weights and return an unnormalised return R
     - return is normalised to be between 0 and 1 by R_norm = (R - r_min) / (r_max - r_min)
     - the context (normal to contor lines of objective) is normalised to the new space by w' = S^-T w
     - where S^-T corresponds to scaling by (r_max - r_min), as S corresponds to scaling by 1 / (r_max - r_min)^-1
     - The normalised weight is then w_norm = w' / 1^T w' = S^-T w / 1^T S^-T w
     - Contextual return is then computed as w_norm^T R_norm
     - Which gives the following after rearrangement:
     - w_norm^T R_norm
     - = (S^-T w)^T R_norm / 1^T S^-T w             (defn w_norm =  S^-T w / 1^T S^-T w)
     - = w^T S^-1 R_norm / 1^T S^-T w               (rearranging)
     - = w^T S^-1 S R / 1^T S^-T w                  (R_norm = S R)
     - = w^T R / 1^T S^-T w                         (simplify)

     So we can compute that as a reweighting of the contextual return, where the reweighting is given by: 
     alpha(w) = 1 / 1^T S^-T w = 1 / dot(r_max - r_min, w)
    */
    class MoMCEvaluator : public MCEvaluator {
        protected:
            std::vector<Vec> mo_sampled_returns;
            std::vector<double> sampled_ctx_returns;
            std::vector<double> sampled_reweighted_ctx_returns;
            Vec r_min;
            Vec r_max;
            bool well_spaced_eval;
            bool normalised_value_space;
            std::vector<Vec> context_weights;

            /**
             * Runs a single rollout and stores the result in 'sampled_returns'.
            */
            virtual void run_rollout(int thread_id, EvalPolicy& thread_policy) override;

            /**
             * Computes a reweighting of the contextual return 
             * alpha(w) = 1 / dot(r_max - r_min, w)
             */
            double reweighting_coefficient(Vec context_weight);



        public:
            MoMCEvaluator(
                std::shared_ptr<EvalPolicy> eval_policy,
                int max_trial_length,
                std::shared_ptr<MoThtsManager> manager,
                Vec r_min,
                Vec r_max,
                bool well_spaced_eval=false,
                bool normalised_value_space=false);
            
            virtual ~MoMCEvaluator() = default;

            /**
             * Override of run rollouts
            */
            virtual void run_rollouts(int num_rollouts, int num_threads) override;

            /**
             * Returns the mean return of 'sampled_returns'
            */
            Vec get_mo_return_mean();
            double get_mo_ctx_return_mean();
            double get_reweighted_mo_ctx_return_mean();

            /**
                * Returns the stddev of 'sampled_returns'
            */
            Vec get_mo_return_variance();
            double get_mo_ctx_return_variance();
            double get_reweighted_mo_ctx_return_variance();
    };
}