#include "mo/smt_bts_decision_node.h"

#include "helper_templates.h"
#include "mo/mo_helper.h"

#include <limits>
#include <sstream>

using namespace std; 

namespace thts {
    SmtBtsDNode::SmtBtsDNode(
        shared_ptr<SmtBtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmtBtsCNode> parent) :
            SmtThtsDNode(
                static_pointer_cast<SmtThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmtThtsCNode>(parent)),
            _action_ctx_key(),
            _ball_ctx_key()
    {
        // todotodo
        stringstream ss_a;
        ss_a << "a_" << decision_depth;
        _action_ctx_key = ss_a.str();
        stringstream ss_b;
        ss_b << "b_" << decision_depth;
        _ball_ctx_key = ss_b.str();
    }
    
    void SmtBtsDNode::visit(MoThtsContext& ctx) 
    {
        num_visits += 1;
    } 

    shared_ptr<const Action> SmtBtsDNode::select_action(MoThtsContext& ctx) 
    {
        // todotodo
        return nullptr;
    }

    shared_ptr<const Action> SmtBtsDNode::recommend_action(MoThtsContext& ctx) const 
    { 
        // todotodo
        return nullptr;
    }

    void SmtBtsDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {
        // todotodo
    }

    string SmtBtsDNode::get_pretty_print_val() const {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmtBtsCNode> SmtBtsDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<SmtBtsCNode>(new_child);
    }

    shared_ptr<SmtThtsCNode> SmtBtsDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<SmtBtsCNode> new_child = make_shared<SmtBtsCNode>(
            static_pointer_cast<SmtBtsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const SmtBtsDNode>(shared_from_this()));
        return static_pointer_cast<SmtThtsCNode>(new_child);
    }

    shared_ptr<SmtBtsCNode> SmtBtsDNode::get_child_node(shared_ptr<const Action> action) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<SmtBtsCNode>(new_child);
    }
}