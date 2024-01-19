#include "mo/czt_decision_node.h"

using namespace std; 

namespace thts {
    CztCNode::CztCNode(
        shared_ptr<CztManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const CztDNode> parent) :
            BL_MoThtsCNode(
                static_pointer_cast<BL_MoThtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const BL_MoThtsDNode>(parent))
    {
    }
    
    void CztCNode::visit(MoThtsContext& ctx) 
    {
        num_visits += 1;
    }  

    shared_ptr<const State> CztCNode::sample_observation(MoThtsContext& ctx) 
    {
        shared_ptr<const Observation> obs = thts_manager->thts_env()->sample_transition_distribution_itfc(
            state, action, *thts_manager, ctx); 
        shared_ptr<const State> next_state = static_pointer_cast<const State>(obs);
        if (!has_child_node_itfc(obs)) {
            create_child_node(next_state);
        }
        return next_state;
    }

    void CztCNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {  
        // Backup handled in decision nodes
    }

    string CztCNode::get_pretty_print_val() const 
    {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<BL_MoThtsDNode> CztCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {
        shared_ptr<CztDNode> new_child = make_shared<CztDNode>(
            static_pointer_cast<CztManager>(thts_manager), 
            next_state, 
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const CztCNode>(shared_from_this()));
        return static_pointer_cast<BL_MoThtsDNode>(new_child);
    }

    shared_ptr<CztDNode> CztCNode::create_child_node(shared_ptr<const State> next_state) 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obs_itfc);
        return static_pointer_cast<CztDNode>(new_child);
    }

    shared_ptr<CztDNode> CztCNode::get_child_node(shared_ptr<const State> next_state) const 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obs_itfc);
        return static_pointer_cast<CztDNode>(new_child);
    }
}