#include "algorithms/idents_chance_node.h"

using namespace std;

namespace thts {
    /**
     * Constructor
     */
    IDentsCNode::IDentsCNode(
        shared_ptr<IDentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const IDentsDNode> parent) :
            IDBDentsCNode(
                thts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const IDBDentsDNode>(parent))
    {
    }

    /**
     * Make child node
     */
    shared_ptr<IDentsDNode> IDentsCNode::create_child_node_helper(
        shared_ptr<const State> observation, shared_ptr<const State> next_state) const 
    {  
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<IDentsDNode>(
            static_pointer_cast<IDentsManager>(ThtsCNode::thts_manager), 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const IDentsCNode>(shared_from_this()));
    }

    /**
     * Return string with all of the relevant values in this node
     */
    string IDentsCNode::get_pretty_print_val() const {
        stringstream ss;
        ss << soft_value << "(dp:" << dp_value << ",e:" << subtree_entropy << ")";
        return ss.str();
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsDNode> IDentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<IDentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}