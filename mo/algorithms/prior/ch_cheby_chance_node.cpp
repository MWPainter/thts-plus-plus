#include "mo/algorithms/prior/ch_cheby_chance_node.h"

#include <sstream>

using namespace std; 

namespace thts {
    ChChebyUctCNode::ChChebyUctCNode(
        shared_ptr<ChChebyUctManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChChebyUctDNode> parent) :
            ChUctCNode(
                static_pointer_cast<ChUctManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ChUctDNode>(parent))
    {
    }

    void ChChebyUctCNode::visit(MoThtsContext& ctx)
    {
        // Call parent visit
        ChUctCNode::visit(ctx);

        // Add running reward to context
        ChChebyUctManager& manager = (ChChebyUctManager&) *thts_manager;
        Vec running_reward = Vec::Zero(manager.reward_dim);
        if (ctx.context_map_contains(RUNNING_REWARD_CTX_KEY)) {
            running_reward = ctx.get_value<Vec>(RUNNING_REWARD_CTX_KEY);
        }
        MoThtsEnv& env = *dynamic_pointer_cast<MoThtsEnv>(manager.thts_env());
        Vec local_reward = env.get_mo_reward_itfc(state,action,ctx);
        ctx.put_value(RUNNING_REWARD_CTX_KEY, make_shared<Vec>(running_reward + local_reward));
    }

    string ChChebyUctCNode::get_pretty_print_val() const 
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
    shared_ptr<ChThtsDNode> ChChebyUctCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {   
        shared_ptr<ChChebyUctDNode> child_node = make_shared<ChChebyUctDNode>(
            static_pointer_cast<ChChebyUctManager>(thts_manager), 
            next_state, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChChebyUctCNode>(shared_from_this()));
        shared_ptr<const Observation> obs = static_pointer_cast<const Observation>(next_state);
        return static_pointer_cast<ChThtsDNode>(child_node);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChChebyUctCNode::visit_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }
    
    shared_ptr<const Observation> ChChebyUctCNode::sample_observation_itfc(ThtsContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        shared_ptr<const State> obs = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obs);
    }

    void ChChebyUctCNode::backup_itfc(
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

    shared_ptr<ThtsDNode> ChChebyUctCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, 
        shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obs_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<ChThtsDNode> child_node = create_child_node_helper(obs_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
} 