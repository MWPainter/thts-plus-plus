#include "py/mo_py_mc_eval.h"

using namespace std;


/**
 * MC Eval implementation
*/
namespace thts::python {
    MoPyMCEvaluator::MoPyMCEvaluator(
        shared_ptr<EvalPolicy> policy, 
        int max_trial_length, 
        shared_ptr<MoThtsManager> manager,
        Eigen::ArrayXd r_min,
        Eigen::ArrayXd r_max) :
            MCEvaluator(policy,max_trial_length,manager),
            MoMCEvaluator(policy,max_trial_length,manager,r_min,r_max),
            PyMCEvaluator(policy,max_trial_length,manager)
    {
    }

    /**
     * Mo overrides
    */
    void MoPyMCEvaluator::run_rollout(int thread_id, EvalPolicy& thread_policy) 
    {
        MoMCEvaluator::run_rollout(thread_id, thread_policy);
    }

    void MoPyMCEvaluator::thread_run_rollouts(
        int total_rollouts, 
        int thread_id, 
        int num_threads, 
        std::shared_ptr<EvalPolicy> thread_policy) 
    {
        PyMCEvaluator::thread_run_rollouts(total_rollouts,thread_id,num_threads,thread_policy);
    }
    
    void MoPyMCEvaluator::run_rollouts(int num_rollouts, int num_threads) 
    {
        PyMCEvaluator::run_rollouts(num_rollouts, num_threads);
    }
     
}