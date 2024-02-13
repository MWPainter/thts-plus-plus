#include "mo/ch_thts_decision_node.h"

#include <sstream>

using namespace std; 

namespace thts {
    CH_MoThtsCNode::CH_MoThtsCNode(
        shared_ptr<CH_MoThtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const CH_MoThtsDNode> parent) :
            MoThtsCNode(
                static_pointer_cast<MoThtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MoThtsDNode>(parent)),
            num_backups(0),
            convex_hull()
    {
    }
    
    void CH_MoThtsCNode::visit(MoThtsContext& ctx) 
    {
        num_visits += 1;
    } 

    void CH_MoThtsCNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx)
    {
        int total_child_backups = 0;
        for (shared_ptr<ThtsDNode> child : children) {
            CH_MoThtsDNode& ch_child = (CH_MoThtsDNode&) *child;
            lock_guard<mutex> lg(ch_child.get_lock());
            total_child_backups += ch_child.num_backups;
        }
        
        convex_hull = ConvexHull<shared_ptr<const Action>>();
        for (shared_ptr<ThtsDNode> child : children) {
            CH_MoThtsDNode& ch_child = (CH_MoThtsDNode&) *child;
            lock_guard<mutex> lg(ch_child.get_lock());
            convex_hull += ch_child.convex_hull * (ch_child.num_backups / total_child_backups);
        }
    }

    string CH_MoThtsCNode::get_convex_hull_pretty_print_string() const {
        stringstream ss;
        ss << convex_hull;
        return ss.str();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<CH_MoThtsDNode> CH_MoThtsCNode::create_child_node(shared_ptr<const State> next_state) 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obs_itfc);
        return static_pointer_cast<CH_MoThtsDNode>(new_child);
    }

    shared_ptr<CH_MoThtsDNode> CH_MoThtsCNode::get_child_node(shared_ptr<const State> next_state) const 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obs_itfc);
        return static_pointer_cast<CH_MoThtsDNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void CH_MoThtsCNode::visit_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        visit(ctx_itfc);
    }
    
    shared_ptr<const Observation> CH_MoThtsCNode::sample_observation_itfc(ThtsEnvContext& ctx) 
    {
        MoThtsContext& ctx_itfc = (MoThtsContext&) ctx;
        shared_ptr<const State> obs = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obs);
    }

    void CH_MoThtsCNode::backup_itfc(
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

    shared_ptr<ThtsDNode> CH_MoThtsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, 
        shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obs_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<CH_MoThtsDNode> child_node = create_child_node_helper(obs_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
} 