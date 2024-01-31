#include "mo/smt_bts_chance_node.h"

using namespace std; 

namespace thts {
    SmtBtsCNode::SmtBtsCNode(
        shared_ptr<SmtBtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmtBtsDNode> parent) :
            SmtThtsCNode(
                static_pointer_cast<SmtThtsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmtThtsDNode>(parent))
    {
    }
    
    void SmtBtsCNode::visit(MoThtsContext& ctx) 
    {
        num_visits += 1;
    }  

    shared_ptr<const State> SmtBtsCNode::sample_observation(MoThtsContext& ctx) 
    {
        // todotodo
    }

    void SmtBtsCNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {  
        // todotodo
    }

    string SmtBtsCNode::get_pretty_print_val() const 
    {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmtThtsDNode> SmtBtsCNode::create_child_node_helper(shared_ptr<const State> next_state) const 
    {
        shared_ptr<SmtBtsDNode> new_child = make_shared<SmtBtsDNode>(
            static_pointer_cast<SmtBtsManager>(thts_manager), 
            next_state, 
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const SmtBtsCNode>(shared_from_this()));
        return static_pointer_cast<SmtThtsDNode>(new_child);
    }

    shared_ptr<SmtBtsDNode> SmtBtsCNode::create_child_node(shared_ptr<const State> next_state) 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obs_itfc);
        return static_pointer_cast<SmtBtsDNode>(new_child);
    }

    shared_ptr<SmtBtsDNode> SmtBtsCNode::get_child_node(shared_ptr<const State> next_state) const 
    {
        shared_ptr<const Observation> obs_itfc = static_pointer_cast<const Observation>(next_state);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obs_itfc);
        return static_pointer_cast<SmtBtsDNode>(new_child);
    }
}