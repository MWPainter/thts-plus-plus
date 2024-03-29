#include "algorithms/ments/dbments_chance_node.h"

using namespace std;

namespace thts {
    /**
     * Constructor
     */
    DBMentsCNode::DBMentsCNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const DBMentsDNode> parent) :
            MentsCNode(
                thts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsDNode>(parent)),
            DPCNode()
    {
    }

    /**
     * Calls ments soft backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void DBMentsCNode::backup(
        const std::vector<double>& trial_rewards_before_node, 
        const std::vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx)
    {   
        backup_soft();
        backup_dp<DBMentsDNode>(children, local_reward, is_opponent());
    }

    /**
     * Make child node
     */
    shared_ptr<DBMentsDNode> DBMentsCNode::create_child_node_helper(
        shared_ptr<const State> observation, shared_ptr<const State> next_state) const 
    {  
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<DBMentsDNode>(
            static_pointer_cast<MentsManager>(ThtsCNode::thts_manager), 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const DBMentsCNode>(shared_from_this()));
    }

    /**
     * Return string of the soft value
     */
    string DBMentsCNode::get_pretty_print_val() const {
        stringstream ss;
        ss << dp_value << "(soft_val:" << soft_value << ")";
        return ss.str();
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsDNode> DBMentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<DBMentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}