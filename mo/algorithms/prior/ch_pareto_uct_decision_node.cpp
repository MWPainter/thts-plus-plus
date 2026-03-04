#include "mo/algorithms/prior/ch_pareto_uct_decision_node.h"

#include "helper_templates.h"

#include <cmath>

using namespace std; 

namespace thts {
    ChParetoUctDNode::ChParetoUctDNode(
        shared_ptr<ChParetoUctManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChParetoUctCNode> parent) :
            ChUctDNode(
                static_pointer_cast<ChUctManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ChUctCNode>(parent))
    {
    }

    double ChParetoUctDNode::compute_ucb_confidence_interval(int num_visits, int child_visits) const {
        MoThtsManager& manager = (MoThtsManager&) *thts_manager;
        double num_visits_d = (num_visits > 0) ? (double)num_visits : 1.0;
        double child_visits_d = (child_visits > 0) ? (double)child_visits : 1.0;
        return sqrt((log(num_visits_d) + log(manager.reward_dim)) / child_visits_d);
    }

    shared_ptr<const Action> ChParetoUctDNode::select_action(MoThtsContext& ctx)
    {
        ChParetoUctManager& manager = (ChParetoUctManager&) *thts_manager;

        // Make sure all arms initialised
        shared_ptr<vector<shared_ptr<const Action>>> env_actions = manager.thts_env()->get_valid_actions_itfc(state, ctx);
        vector<shared_ptr<const Action>> actions_yet_to_try;
        for (shared_ptr<const Action> action : *env_actions) {
            if (!has_child_node_itfc(action)) {
                actions_yet_to_try.push_back(action);
            }
        }

        if (actions_yet_to_try.size() > 0) {
            int indx = manager.get_rand_int(0,actions_yet_to_try.size());
            shared_ptr<const Action> action = actions_yet_to_try[indx];
            create_child_node(action);
            return action;
        }

        vector<shared_ptr<const Action>> actions_to_consider = this->get_actions_to_consider(ctx);
        unordered_set<shared_ptr<const Action>> actions_to_consider_set(actions_to_consider.begin(), actions_to_consider.end());

        // Compute convex hull from children with confidence interval terms
        // Keeping track of which actions could lead to each value
        unordered_map<Vec,vector<shared_ptr<const Action>>> vec_to_action_map;
        int local_visits = get_num_visits(ctx);
        ConvexHull pareto_ch;
        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pair : children) {
            shared_ptr<const Action> action = pair.first;
            if (!actions_to_consider_set.contains(action)) {
                continue;
            }
            ChParetoUctCNode& child = (ChParetoUctCNode&) *pair.second;
            int child_visits = child.get_num_visits(ctx);
            Eigen::ArrayXd ucb_conf_vec = Eigen::ArrayXd::Ones(manager.reward_dim) * compute_ucb_confidence_interval(local_visits, child_visits);
            ConvexHull shifted_child_ch = child.convex_hull_for_search + Vec(ucb_conf_vec);
            for (const Vec& v : shifted_child_ch.ch_points) {
                vec_to_action_map[v].push_back(action);
            }
            pareto_ch |= shifted_child_ch;
        }

        // convex pareto_ch to a list of Vec objects, and sample from it randomly
        // small chance in multi-threaded environments that the convex hull is empty, so we sample action randomly from all actions in that case
        vector<Vec> pareto_ch_points;
        for (const Vec& v : pareto_ch.ch_points) {
            pareto_ch_points.push_back(v);
        }
        if (pareto_ch_points.size() == 0) {
            int act_indx = manager.get_rand_int(0,env_actions->size());
            return env_actions->at(act_indx);
        }
        int vec_indx = manager.get_rand_int(0,pareto_ch_points.size());
        Vec random_ch_point = pareto_ch_points[vec_indx];

        // Sample an action randomly from the list of actions that lead to that node
        vector<shared_ptr<const Action>>& vec_actions = vec_to_action_map[random_ch_point];
        int act_indx = manager.get_rand_int(0,vec_actions.size());
        return vec_actions[act_indx];
    }

    string ChParetoUctDNode::get_pretty_print_val() const 
    {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    /**
     * Added making the child's czt_node pointing to the same CztCNode as our czt_node
    */
    shared_ptr<ChThtsCNode> ChParetoUctDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<ChParetoUctCNode> child_node = make_shared<ChParetoUctCNode>(
            static_pointer_cast<ChParetoUctManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChParetoUctDNode>(shared_from_this()));
        return static_pointer_cast<ChThtsCNode>(child_node);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChParetoUctDNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> ChParetoUctDNode::select_action_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return select_action(mo_ctx);
    }

    shared_ptr<const Action> ChParetoUctDNode::recommend_action_itfc(ThtsContext& ctx) const 
    {
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        return recommend_action(mo_ctx);
    }

    void ChParetoUctDNode::backup_itfc(
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

    shared_ptr<ThtsCNode> ChParetoUctDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ChThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}