#include "algorithms/dents_chance_node.h"

using namespace std;

namespace thts {
    /**
     * Constructor
     */
    DentsCNode::DentsCNode(
        shared_ptr<DentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const DentsDNode> parent) :
            MentsCNode(
                thts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsDNode>(parent))
    {
    }

    /**
     * Make child node
     */
    shared_ptr<DentsDNode> DentsCNode::create_child_node_helper(
        shared_ptr<const State> observation, shared_ptr<const State> next_state) const 
    {  
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<DentsDNode>(
            static_pointer_cast<DentsManager>(thts_manager), 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const DentsCNode>(shared_from_this()));
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsDNode> DentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<DentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}