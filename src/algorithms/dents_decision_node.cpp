#include "algorithms/dents_decision_node.h"

#include "algorithms/common/decaying_temp.h"

using namespace std; 

namespace thts {
    DentsDNode::DentsDNode(
        shared_ptr<DentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const DentsCNode> parent) :
            MentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsCNode>(parent))
    {
    }

    /**
     * Call decaying temp fn when getting temp
     */
    double DentsDNode::get_temp() const {
        DentsManager& manager = (DentsManager&) *thts_manager;
        return get_decayed_temp(manager.temp, num_visits, manager.min_temp);
    }

    /**
     * Return string of the soft value
     */
    string DentsDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << soft_value << "(t:" << get_temp() << ")";
        return ss.str();
    }

    /**
     * Make child node
     */
    shared_ptr<DentsCNode> DentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<DentsCNode>(
            static_pointer_cast<DentsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const DentsDNode>(shared_from_this()));
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsCNode> DentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<DentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}