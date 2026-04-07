#include "algorithms/uct/max_uct_decision_node.h"

#include "helper_templates.h"

#include <cmath>
#include <float.h>
#include <sstream>
#include <vector>

using namespace std; 

namespace thts {
    /**
     * Constructor, inits members. 
     * 
     * If we have a heuristic function, then initialises 'num_visits' and 'avg_return' according to the heuristic in 
     * the manager. If we have a prior function, then
     */
    MaxUctDNode::MaxUctDNode(
        shared_ptr<UctManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MaxUctCNode> parent) :
            UctDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const UctCNode>(parent))
    {  
        if (thts_manager->heuristic_fn != nullptr 
            && !thts_manager->thts_env()->is_sink_state_itfc(state,*thts_manager->get_thts_context())) 
        {
            avg_return_local = heuristic_value; 
        }  
    }

    /**
     * Fill q values to use for search
     */
    void MaxUctDNode::fill_q_values(unordered_map<shared_ptr<const Action>,double>& q_values) const {
        for (shared_ptr<const Action> action : *actions) {
            q_values[action] = get_child_node(action)->avg_return_local;
        }
    }

    /**
     * Calls the running average return backup function.
     */
    void MaxUctDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsContext& ctx) 
    {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        avg_return = opp_coeff * -numeric_limits<double>::infinity();
        avg_return_local = opp_coeff * -numeric_limits<double>::infinity();

        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pr : children) {
            MaxUctCNode& child = (MaxUctCNode&) *pr.second;
            if (child.num_backups == 0) continue;
            if (opp_coeff * child.avg_return > opp_coeff * avg_return) {
                avg_return = child.avg_return;
                avg_return_local = child.avg_return;
            }
        }

        num_backups++;

        // mix in heuristic value if we have one
        if (has_heuristic_value()) 
        {
            double effective_num_backups = num_backups + thts_manager->heuristic_weight_global;
            avg_return *= num_backups / effective_num_backups;
            avg_return += thts_manager->heuristic_weight_global * heuristic_value / effective_num_backups;

            avg_return_local = avg_return;
            effective_num_backups += thts_manager->heuristic_weight_local;
            avg_return_local *= num_backups / effective_num_backups;
            avg_return_local += thts_manager->heuristic_weight_local * heuristic_value / effective_num_backups;
        }
    }
    
    /**
     * Make a child
     */
    shared_ptr<MaxUctCNode> MaxUctDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<MaxUctCNode>(
            static_pointer_cast<UctManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const MaxUctDNode>(shared_from_this()));
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<MaxUctCNode> MaxUctDNode::create_child_node(shared_ptr<const Action> action) {
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(action);
        return static_pointer_cast<MaxUctCNode>(new_child);
    }

    bool MaxUctDNode::has_child_node(shared_ptr<const Action> action) const {
        return ThtsDNode::has_child_node_itfc(static_pointer_cast<const Action>(action));
    }

    shared_ptr<MaxUctCNode> MaxUctDNode::get_child_node(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<MaxUctCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void MaxUctDNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsContext& ctx)
    {
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx);
    }

    shared_ptr<ThtsCNode> MaxUctDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<MaxUctCNode> child_node = create_child_node_helper(action);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}
