#include "mo/mo_thts_chance_node.h"

using namespace std;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     */
    MoThtsCNode::MoThtsCNode(
        shared_ptr<MoThtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MoThtsDNode> parent) :
            ThtsCNode(thts_manager, state, action, decision_depth, decision_timestep, parent)
    {
    }
}