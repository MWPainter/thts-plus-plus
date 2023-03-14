#include "algorithms/esthts_decision_node.h"

using namespace std; 

namespace thts {
    /**
     * Constructor
    */
    ESThtsDNode::ESThtsDNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ESThtsCNode> parent) :
            DBMentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DBMentsCNode>(parent))
    {
    }

    /**
     * Gets the q_value to use for a child, calls ments version when there is not a child node
     */
    double ESThtsDNode::get_soft_q_value(std::shared_ptr<const Action> action, double opp_coeff) const {
        if (has_child_node(action)) {
            ESThtsCNode& child = (ESThtsCNode&) *get_child_node(action);
            return child.dp_value * opp_coeff;
        } 

        return MentsDNode::get_soft_q_value(action, opp_coeff);
    }

    /**
     * Calls both the ments soft backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void ESThtsDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        backup_dp<DBMentsCNode>(children, is_opponent());
    }

    /**
     * Return string of the soft value
     */
    string ESThtsDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << dp_value;
        return ss.str();
    }

    /**
     * Make child node
     */
    shared_ptr<ESThtsCNode> ESThtsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<ESThtsCNode>(
            static_pointer_cast<MentsManager>(ThtsDNode::thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const ESThtsDNode>(shared_from_this()));
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsCNode> ESThtsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ESThtsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}