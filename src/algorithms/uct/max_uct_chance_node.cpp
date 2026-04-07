#include "algorithms/uct/max_uct_chance_node.h"

#include "helper_templates.h"

using namespace std; 

namespace thts {
    /**
     * Construct Uct Chance node. Use thts_manager to initialise values with heuristic if necessary.
     */
    MaxUctCNode::MaxUctCNode(
        shared_ptr<UctManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MaxUctDNode> parent) :
            UctCNode(
                thts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const UctDNode>(parent))
    {  
    }

    /**
     * Calls the running average return backup function.
     */
    void MaxUctCNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsContext& ctx) 
    {
        // backup_average_return(trial_cumulative_return_after_node);
        
        avg_return = 0.0;
        avg_return_local = 0.0;
        double sum_child_n_selections = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<ThtsDNode>> pr : children) {
            shared_ptr<const Observation> observation = pr.first;
            MaxUctDNode& child = (MaxUctDNode&) *pr.second;
            double child_n_selections = empirical_distribution[observation];
            if (child_n_selections == 0) continue;
            sum_child_n_selections += child_n_selections;
            avg_return *= (sum_child_n_selections - child_n_selections) / sum_child_n_selections;
            avg_return += child_n_selections * child.avg_return / sum_child_n_selections;
            avg_return_local *= (sum_child_n_selections - child_n_selections) / sum_child_n_selections;
            avg_return_local += child_n_selections * child.avg_return_local / sum_child_n_selections;
        }
        avg_return += local_reward; // +R(s,a)
        avg_return_local += local_reward; // +R(s,a)

        num_backups++;
    }

    shared_ptr<MaxUctDNode> MaxUctCNode::create_child_node_helper(shared_ptr<const State> observation) const
    {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        return make_shared<MaxUctDNode>(
            static_pointer_cast<UctManager>(thts_manager), 
            next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const MaxUctCNode>(shared_from_this()));
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<MaxUctDNode> MaxUctCNode::create_child_node(shared_ptr<const State> observation) 
    {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obsv_itfc);
        return static_pointer_cast<MaxUctDNode>(new_child);
    }

    bool MaxUctCNode::has_child_node(std::shared_ptr<const State> observation) const {
        return ThtsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
    }
    
    shared_ptr<MaxUctDNode> MaxUctCNode::get_child_node(shared_ptr<const State> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<MaxUctDNode>(new_child);
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {

    void MaxUctCNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsContext& ctx) 
    {
        ThtsContext& ctx_itfc = (ThtsContext&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<ThtsDNode> MaxUctCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<MaxUctDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}