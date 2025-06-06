#include "mo/algorithms/chmcts/ch_uct_decision_node.h"

#include "helper_templates.h"

using namespace std; 

namespace thts {
    ChUctDNode::ChUctDNode(
        shared_ptr<ChUctManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChUctCNode> parent) :
            ChThtsDNode(
                static_pointer_cast<ChThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ChThtsCNode>(parent))
    {
    }

    double ChUctDNode::compute_ucb_confidence_interval(int num_visits, int child_visits) const {
        double num_visits_d = (num_visits > 0) ? (double)num_visits : 1.0;
        double child_visits_d = (child_visits > 0) ? (double)child_visits : 1.0;
        return sqrt(log(num_visits_d) / child_visits_d);
    }

    void ChUctDNode::fill_ucb_q_values(ActionDistr& ucb_q_values, MoThtsContext& ctx) const
    {
        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pair : children) {
            shared_ptr<const Action> action = pair.first;
            ChUctCNode& child = (ChUctCNode&) *get_child_node(action);
            lock_guard<mutex> lg(child.node_lock);
            ucb_q_values[action] = child.get_contextual_q_value(ctx);
        }
    }

    void ChUctDNode::fill_ucb_values(ActionDistr& ucb_values, MoThtsContext& ctx) const 
    {
        ChUctManager& manager = (ChUctManager&) *thts_manager;
        ThtsEnv& env = *manager.thts_env();

        // Compute Q-values to use with ucb
        ActionDistr ucb_q_values;
        fill_ucb_q_values(ucb_q_values, ctx);

        // Compute adaptive bias if using
        double bias = manager.bias; 
        if (manager.adaptive_bias) {
            double adaptive_bias_coef = bias;
            bias = ChUctManager::ADAPTIVE_BIAS_MIN_BIAS;
            for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pair : children) {
                shared_ptr<const Action> action = pair.first;
                ChUctCNode& child = (ChUctCNode&) *get_child_node(action);
                lock_guard<mutex> lg(child.node_lock);
                double child_abs_val = abs(ucb_q_values[action]);
                double candidate_bias = child_abs_val * adaptive_bias_coef;
                if (candidate_bias > bias) bias = candidate_bias;
            }
        }

        // Compute ucb values
        shared_ptr<ActionVector> actions = env.get_valid_actions_itfc(state,ctx);
        int local_visits = get_num_visits(ctx);
        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node_itfc(action)) {
                ucb_values[action] = numeric_limits<double>::max();
                continue;
            }
            ChUctCNode& child = (ChUctCNode&) *get_child_node(action);
            child.node_lock.lock();
            int child_visits = get_child_node(action)->get_num_visits(ctx);
            child.node_lock.unlock();
            double action_ucb_value = compute_ucb_confidence_interval(local_visits, child_visits);
            action_ucb_value *= bias;
            // if (has_prior()) {
            //     action_ucb_value *= policy_prior->at(action);
            // }
            action_ucb_value += ucb_q_values[action];
            ucb_values[action] = action_ucb_value;
        }  
    }

    shared_ptr<const Action> ChUctDNode::select_action(MoThtsContext& ctx)
    {
        unordered_map<shared_ptr<const Action>,double> ucb_values;
        fill_ucb_values(ucb_values, ctx);
        shared_ptr<const Action> selected_act = helper::get_max_key_break_ties_randomly(ucb_values, *thts_manager);
        if (!has_child_node_itfc(selected_act)) {
            create_child_node(selected_act);
        }
        return selected_act;
    }

    string ChUctDNode::get_pretty_print_val() const 
    {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<ChUctCNode> ChUctDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<ChUctCNode>(new_child);
    } 

    /**
     * Added making the child's czt_node pointing to the same CztCNode as our czt_node
    */
    shared_ptr<ChThtsCNode> ChUctDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<ChUctCNode> child_node = make_shared<ChUctCNode>(
            static_pointer_cast<ChUctManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChUctDNode>(shared_from_this()));
        return static_pointer_cast<ChThtsCNode>(child_node);
    }

    shared_ptr<ChUctCNode> ChUctDNode::get_child_node(shared_ptr<const Action> action) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<ChUctCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChUctDNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> ChUctDNode::select_action_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> ChUctDNode::recommend_action_itfc(ThtsContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void ChUctDNode::backup_itfc(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<ThtsCNode> ChUctDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ChThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}