#include "algorithms/idents_decision_node.h"

using namespace std; 

namespace thts {
    IDentsDNode::IDentsDNode(
        shared_ptr<IDentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const IDentsCNode> parent) :
            IDBDentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const IDBDentsCNode>(parent))
    {
    }
    
    /**
     * IDents just needs to switch out the recommendation function from the DP recommendation to MENTS recommendation
    */
    shared_ptr<const Action> IDentsDNode::recommend_action(ThtsEnvContext& ctx) const {
        return recommend_action_best_soft_value();
    }

    /**
     * Make child node
     */
    shared_ptr<IDentsCNode> IDentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<IDentsCNode>(
            static_pointer_cast<IDentsManager>(ThtsDNode::thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const IDentsDNode>(shared_from_this()));
    }

    /**
     * Return string with all of the relevant values in this node
     */
    string IDentsDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << soft_value << "(dp:" << dp_value << ",e:" << subtree_entropy << ",t:" << get_decayed_temp() << ")";
        return ss.str();
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsCNode> IDentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<IDentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}