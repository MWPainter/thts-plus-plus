#include "multi_objective/mo_thts.h"

#include "thts_types.h"
#include "multi_objective/mo_thts_chance_node.h"

#include <utility>

using namespace std;


namespace thts {
    /**
     * Constructor.
     */
    MoThtsPool::MoThtsPool(
        shared_ptr<ThtsManager> thts_manager, 
        shared_ptr<MoThtsDNode> root_node, 
        int num_threads, 
        shared_ptr<ThtsLogger> logger) :
            ThtsPool(thts_manager, root_node, num_threads, logger)
    {
    }

    /**
     * See ThtsPool::run_selection_phase
     * 
<<<<<<< HEAD
     * This is a copy and pase, replacing the reward type from double to Eigen::ArrayXd, and calling 
=======
     * This is a copy and pase, replacing the reward type from double to Eigen::VectorXd, and calling 
>>>>>>> ea559dd96862f49bddc1102d6f908fe2e2755913
     * 'get_mo_reward_itfc' rather than 'get_reward_itfc', and using mo_heuristic_value instead of heuristic_value
     */
    void MoThtsPool::run_selection_phase(
        vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>>& nodes_to_backup, 
<<<<<<< HEAD
        vector<Eigen::ArrayXd>& rewards, 
=======
        vector<Eigen::VectorXd>& rewards, 
>>>>>>> ea559dd96862f49bddc1102d6f908fe2e2755913
        ThtsEnvContext& context)
    {
        bool new_decision_node_created_this_trial = false;
        shared_ptr<ThtsDNode> cur_node = root_node;

        while (should_continue_selection_phase(cur_node, new_decision_node_created_this_trial)) {
            // dnode visit + select action
            cur_node->lock();
            cur_node->visit_itfc(context);
            shared_ptr<const Action> action = cur_node->select_action_itfc(context);
            shared_ptr<ThtsCNode> chance_node = cur_node->get_child_node_itfc(action);
            cur_node->unlock();
            
            // cnode visit + sample outcome
            chance_node->lock();
            int pre_visit_children = chance_node->get_num_children();
            chance_node->visit_itfc(context);
            shared_ptr<const Observation> observation = chance_node->sample_observation_itfc(context);
            int post_visit_children = chance_node->get_num_children();
            if (post_visit_children > pre_visit_children) {
                new_decision_node_created_this_trial = true;
            }
            shared_ptr<ThtsDNode> decision_node = chance_node->get_child_node_itfc(observation);
            chance_node->unlock();

            // push onto 'nodes_to_backup' and 'rewards'
            MoThtsDNode& mo_cur_node = (MoThtsDNode&) *cur_node;
            shared_ptr<const State> state = mo_cur_node.state;
            MoThtsEnv& mo_thts_env = (MoThtsEnv&) *thts_manager->thts_env;
<<<<<<< HEAD
            Eigen::ArrayXd reward = mo_thts_env.get_mo_reward_itfc(state, action, observation);
=======
            Eigen::VectorXd reward = mo_thts_env.get_mo_reward_itfc(state, action, observation);
>>>>>>> ea559dd96862f49bddc1102d6f908fe2e2755913
            nodes_to_backup.push_back(make_pair(cur_node, chance_node));
            rewards.push_back(reward);

            cur_node = decision_node;
        }

        // visit the final node and add heuristic value to list of rewards at end
        cur_node->lock();
        cur_node->visit_itfc(context);
        MoThtsDNode& mo_cur_node = (MoThtsDNode&) *cur_node;
        rewards.push_back(mo_cur_node.mo_heuristic_value);
        cur_node->unlock();
    }


    /**
     * See ThtsPool::run_backup_phase
     * 
<<<<<<< HEAD
     * This is a copy and pase, replacing the reward type from double to Eigen::ArrayXd
=======
     * This is a copy and pase, replacing the reward type from double to Eigen::VectorXd
>>>>>>> ea559dd96862f49bddc1102d6f908fe2e2755913
     * 
     * Added insta return if nothing to backup, and using rewards[0] to get the dimension of the rewards
     */
    void MoThtsPool::run_backup_phase(
        vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>>& nodes_to_backup, 
<<<<<<< HEAD
        vector<Eigen::ArrayXd>& rewards, 
=======
        vector<Eigen::VectorXd>& rewards, 
>>>>>>> ea559dd96862f49bddc1102d6f908fe2e2755913
        ThtsEnvContext& context)
    {
        if (nodes_to_backup.size() == 0) return;

        int dim = rewards[0].rows();
<<<<<<< HEAD
        Eigen::ArrayXd total_return = Eigen::ArrayXd::Constant(dim, 0.0);
        for (Eigen::ArrayXd& reward : rewards) total_return += reward;

        vector<Eigen::ArrayXd> rewards_after;
        vector<Eigen::ArrayXd> rewards_before(rewards);

        Eigen::ArrayXd heuristic_val_at_end = rewards_before.back();
        rewards_before.pop_back();
        rewards_after.push_back(heuristic_val_at_end);

        Eigen::ArrayXd total_return_after = heuristic_val_at_end;

        while (nodes_to_backup.size() > 0) {
            Eigen::ArrayXd reward = rewards_before.back();
=======
        Eigen::VectorXd total_return = Eigen::VectorXd::Constant(dim, 0.0);
        for (Eigen::VectorXd& reward : rewards) total_return += reward;

        vector<Eigen::VectorXd> rewards_after;
        vector<Eigen::VectorXd> rewards_before(rewards);

        Eigen::VectorXd heuristic_val_at_end = rewards_before.back();
        rewards_before.pop_back();
        rewards_after.push_back(heuristic_val_at_end);

        Eigen::VectorXd total_return_after = heuristic_val_at_end;

        while (nodes_to_backup.size() > 0) {
            Eigen::VectorXd reward = rewards_before.back();
>>>>>>> ea559dd96862f49bddc1102d6f908fe2e2755913
            rewards_before.pop_back();
            rewards_after.push_back(reward);
            total_return_after += reward;

            pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>> pr = nodes_to_backup.back();
            MoThtsDNode& decision_node = (MoThtsDNode&) *pr.first;
            MoThtsCNode& chance_node = (MoThtsCNode&) *pr.second;
            nodes_to_backup.pop_back();

            chance_node.lock();
            chance_node.backup_itfc(rewards_before, rewards_after, total_return_after, total_return, context);
            chance_node.unlock();

            decision_node.lock();
            decision_node.backup_itfc(rewards_before, rewards_after, total_return_after, total_return, context);
            decision_node.unlock();
        }
    }

    /**
     * See ThtsPool::run_thts_trial
     * 
<<<<<<< HEAD
     * This is a copy and pase, replacing the reward type from double to Eigen::ArrayXd
     */
    void MoThtsPool::run_thts_trial(int trials_remaining) {
        vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>> nodes_to_backup;
        vector<Eigen::ArrayXd> rewards; 
=======
     * This is a copy and pase, replacing the reward type from double to Eigen::VectorXd
     */
    void MoThtsPool::run_thts_trial(int trials_remaining) {
        vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>> nodes_to_backup;
        vector<Eigen::VectorXd> rewards; 
>>>>>>> ea559dd96862f49bddc1102d6f908fe2e2755913
        
        MoThtsDNode& mo_root_node = (MoThtsDNode&) *root_node;
        shared_ptr<ThtsEnvContext> context = thts_manager->thts_env->sample_context_itfc(mo_root_node.state);
        run_selection_phase(nodes_to_backup, rewards, *context);
        run_backup_phase(nodes_to_backup, rewards, *context);

        try_log();
    }
}