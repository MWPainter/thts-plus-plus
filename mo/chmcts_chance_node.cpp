#include "mo/chmcts_chance_node.h"

#include <sstream>

using namespace std; 

namespace thts {
    ChmctsCNode::ChmctsCNode(
        shared_ptr<ChmctsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ChmctsDNode> parent) :
            CH_MoThtsCNode(
                static_pointer_cast<CH_MoThtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const CH_MoThtsDNode>(parent)),
            czt_node(
                make_shared<CztCNode>(
                    static_pointer_cast<CztManager>(thts_manager),
                    state,
                    action,
                    decision_depth,
                    decision_timestep)) // not passing parent pointer because CZT doesnt use it
    {
    }
    
    void ChmctsCNode::visit(MoThtsContext& ctx) 
    {
        CH_MoThtsCNode::visit(ctx);
        czt_node->visit(ctx);
    } 

    shared_ptr<const State> ChmctsCNode::sample_observation(MoThtsContext& ctx)
    {
        shared_ptr<const State> next_state = czt_node->sample_observation(ctx);
        if (!has_child_node_itfc(static_pointer_cast<const Observation>(next_state))) {
            create_child_node(next_state);
        }
        return next_state;
    }

    void ChmctsCNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx)
    {
        CH_MoThtsCNode::backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return,
            ctx);
        czt_node->backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return,
            ctx);
    }

    string ChmctsCNode::get_pretty_print_val() const 
    {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<ChmctsDNode> ChmctsCNode::create_child_node(shared_ptr<const State> next_state) 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obs_itfc);
        return static_pointer_cast<ChmctsDNode>(new_child);
    }

    /**
     * Added making the child's czt_node pointing to the same CztNode as our czt_node
    */
    shared_ptr<CH_MoThtsDNode> ChmctsCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {   
        shared_ptr<ChmctsDNode> child_node = make_shared<ChmctsDNode>(
            static_pointer_cast<ChmctsManager>(thts_manager), 
            next_state, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ChmctsCNode>(shared_from_this()));
        shared_ptr<const Observation> obs = static_pointer_cast<const Observation>(next_state);
        child_node->czt_node = static_pointer_cast<CztDNode>(czt_node->get_child_node_itfc(obs));
        return static_pointer_cast<CH_MoThtsDNode>(child_node);
    }


    shared_ptr<ChmctsDNode> ChmctsCNode::get_child_node(shared_ptr<const State> next_state) const 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obs_itfc);
        return static_pointer_cast<ChmctsDNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void ChmctsCNode::visit_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }
    
    shared_ptr<const Observation> ChmctsCNode::sample_observation_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        shared_ptr<const State> obs = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obs);
    }

    void ChmctsCNode::backup_itfc(
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

    shared_ptr<ThtsDNode> ChmctsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, 
        shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obs_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<CH_MoThtsDNode> child_node = create_child_node_helper(obs_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
} 