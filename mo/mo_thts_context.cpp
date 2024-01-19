#include "mo/mo_thts_context.h"

#include <iostream>

using namespace std;

namespace thts {

    MoThtsContext::MoThtsContext(MoThtsManager& manager) : context_weight() 
    {
        sample_uniform_random_simplex_for_weight(manager);
    } 

    void MoThtsContext::sample_uniform_random_simplex_for_weight(MoThtsManager& manager)
    {
        context_weight = Eigen::ArrayXd(manager.reward_dim);
        double exp_rvs_run_sum[manager.reward_dim];
        double exp_rvs_sum = 0.0; 
        for (int i=0; i<manager.reward_dim; i++) {
            double exp_rv = manager.get_rand_exp();
            exp_rvs_sum += exp_rv;
            exp_rvs_run_sum[i] = exp_rvs_sum;
        }
        double prev_run_sum = 0.0;
        for (int i=0; i<manager.reward_dim; i++) {
            context_weight[i] = (exp_rvs_run_sum[i] - prev_run_sum) / exp_rvs_sum;
            prev_run_sum = exp_rvs_run_sum[i];
        }
    } 
}