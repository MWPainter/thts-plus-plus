#include "mo/smt_chance_node.h"

using namespace std; 

namespace thts {
    /**
     * TODO: feels like it should be zeros here (for initial value of simplex map), but was having issues with costs
     *  - can reproduce it being a bit weird by setting value to be zeros again in the test env in module.py
    */
    SmtThtsCNode::SmtThtsCNode(
        shared_ptr<SmtThtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmtThtsDNode> parent) :
            MoThtsCNode(
                static_pointer_cast<MoThtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MoThtsDNode>(parent)),
            // simplex_map(thts_manager->reward_dim, Eigen::ArrayXd::Zero(thts_manager->reward_dim))
            simplex_map(thts_manager->reward_dim, thts_manager->default_q_value)
    {
    }
    
    void SmtThtsCNode::visit(MoThtsContext& ctx) 
    {
        num_visits += 1;
    } 

    string SmtThtsCNode::get_simplex_map_pretty_print_string() const {
        return simplex_map.get_pretty_print_string();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmtThtsDNode> SmtThtsCNode::create_child_node(shared_ptr<const State> next_state) 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obs_itfc);
        return static_pointer_cast<SmtThtsDNode>(new_child);
    }

    shared_ptr<SmtThtsDNode> SmtThtsCNode::get_child_node(shared_ptr<const State> next_state) const 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obs_itfc);
        return static_pointer_cast<SmtThtsDNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void SmtThtsCNode::visit_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }
    
    shared_ptr<const Observation> SmtThtsCNode::sample_observation_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        shared_ptr<const State> obs = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obs);
    }

    void SmtThtsCNode::backup_itfc(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<ThtsDNode> SmtThtsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, 
        shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obs_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<SmtThtsDNode> child_node = create_child_node_helper(obs_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
} 