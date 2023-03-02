#include "algorithms/dbdents_decision_node.h"

#include "algorithms/common/decaying_temp.h"

using namespace std; 

namespace thts {
    DBDentsDNode::DBDentsDNode(
        shared_ptr<DentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const DBDentsCNode> parent) :
            DBMentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DBMentsCNode>(parent))
    {
    }

    /**
     * Call decaying temp fn when getting temp
     */
    double DBDentsDNode::get_temp() const {
        DentsManager& manager = (DentsManager&) *ThtsDNode::thts_manager;
        return compute_decayed_temp(manager.temp, num_visits, manager.min_temp);
    }

    /**
     * Return string of the soft value
     */
    string DBDentsDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << dp_value << "(s: " << soft_value << ",t:" << get_temp() << ")";
        return ss.str();
    }

    /**
     * Make child node
     */
    shared_ptr<DBDentsCNode> DBDentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<DBDentsCNode>(
            static_pointer_cast<DentsManager>(ThtsDNode::thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const DBDentsDNode>(shared_from_this()));
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsCNode> DBDentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<DBDentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}