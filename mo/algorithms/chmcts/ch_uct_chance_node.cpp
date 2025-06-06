#include "mo/algorithms/chmcts/ch_uct_chance_node.h"

#include <sstream>

using namespace std; 

namespace thts {
    ChUctCNode::ChUctCNode(
        shared_ptr<ChUctManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChUctDNode> parent) :
            ChThtsCNode(
                static_pointer_cast<ChThtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ChThtsDNode>(parent))
    {
    }

    shared_ptr<const State> ChUctCNode::sample_observation(MoThtsContext& ctx)
    {
        shared_ptr<const Observation> obs = thts_manager->thts_env()->sample_transition_distribution_itfc(
            state, action, *thts_manager, ctx); 
        shared_ptr<const State> next_state = static_pointer_cast<const State>(obs);
        if (!has_child_node_itfc(obs)) {
            create_child_node(next_state);
        }
        return next_state;
    }

    string ChUctCNode::get_pretty_print_val() const 
    {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<ChUctDNode> ChUctCNode::create_child_node(shared_ptr<const State> next_state) 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obs_itfc);
        return static_pointer_cast<ChUctDNode>(new_child);
    }

    /**
     * Added making the child's czt_node pointing to the same CztNode as our czt_node
    */
    shared_ptr<ChThtsDNode> ChUctCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {   
        shared_ptr<ChUctDNode> child_node = make_shared<ChUctDNode>(
            static_pointer_cast<ChUctManager>(thts_manager), 
            next_state, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChUctCNode>(shared_from_this()));
        shared_ptr<const Observation> obs = static_pointer_cast<const Observation>(next_state);
        return static_pointer_cast<ChThtsDNode>(child_node);
    }


    shared_ptr<ChUctDNode> ChUctCNode::get_child_node(shared_ptr<const State> next_state) const 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obs_itfc);
        return static_pointer_cast<ChUctDNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChUctCNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }
    
    shared_ptr<const Observation> ChUctCNode::sample_observation_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        shared_ptr<const State> obs = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obs);
    }

    void ChUctCNode::backup_itfc(
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

    shared_ptr<ThtsDNode> ChUctCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, 
        shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obs_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<ChThtsDNode> child_node = create_child_node_helper(obs_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
} 