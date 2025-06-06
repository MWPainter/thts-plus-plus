#include "mo/algorithms/prior/ch_hvuct_chance_node.h"

#include <sstream>

using namespace std; 

namespace thts {
    ChHvUctCNode::ChHvUctCNode(
        shared_ptr<ChHvUctManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChHvUctDNode> parent) :
            ChUctCNode(
                static_pointer_cast<ChUctManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ChUctDNode>(parent))
    {
    }

    string ChHvUctCNode::get_pretty_print_val() const 
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
     * Added making the child's czt_node pointing to the same CztNode as our czt_node
    */
    shared_ptr<ChThtsDNode> ChHvUctCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {   
        shared_ptr<ChHvUctDNode> child_node = make_shared<ChHvUctDNode>(
            static_pointer_cast<ChHvUctManager>(thts_manager), 
            next_state, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChHvUctCNode>(shared_from_this()));
        shared_ptr<const Observation> obs = static_pointer_cast<const Observation>(next_state);
        return static_pointer_cast<ChThtsDNode>(child_node);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChHvUctCNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }
    
    shared_ptr<const Observation> ChHvUctCNode::sample_observation_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        shared_ptr<const State> obs = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obs);
    }

    void ChHvUctCNode::backup_itfc(
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

    shared_ptr<ThtsDNode> ChHvUctCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, 
        shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obs_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<ChThtsDNode> child_node = create_child_node_helper(obs_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
} 