#include "mo/mo_thts_context.h"

#include <iostream>

using namespace std;

namespace thts {

    MoThtsContext::MoThtsContext(MoThtsManager& manager) : 
        ThtsEnvContext(), 
        context_weight(MoThtsContext::sample_uniform_random_simplex_for_weight(manager)) 
    {
    } 

    MoThtsContext::MoThtsContext(Vec weight) : 
        ThtsEnvContext(), 
        context_weight(weight) 
    {
    } 

    /**
     * https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
     */
    Eigen::ArrayXd MoThtsContext::sample_uniform_random_simplex_for_weight(MoThtsManager& manager)
    {
        Eigen::ArrayXd sampled_weight = Eigen::ArrayXd(manager.reward_dim);
        double exp_rvs_run_sum[manager.reward_dim];
        double exp_rvs_sum = 0.0; 
        for (int i=0; i<manager.reward_dim; i++) {
            double exp_rv = manager.get_rand_exp();
            exp_rvs_sum += exp_rv;
            exp_rvs_run_sum[i] = exp_rvs_sum;
        }
        double prev_run_sum = 0.0;
        for (int i=0; i<manager.reward_dim; i++) {
            sampled_weight[i] = (exp_rvs_run_sum[i] - prev_run_sum) / exp_rvs_sum;
            prev_run_sum = exp_rvs_run_sum[i];
        }
        return sampled_weight;
    } 
}
