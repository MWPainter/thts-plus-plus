#include "algorithms/uct/alphago_chance_node.h"

using namespace std; 

namespace thts {
    /**
     * Construct AlphaGo Chance node
     */
    AlphaGoCNode::AlphaGoCNode(
        shared_ptr<AlphaGoManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const AlphaGoDNode> parent) :
            PuctCNode(thts_manager, state, action, decision_depth, decision_timestep, parent)
    {  
    }

    /**
     * Make a new AlphaGoDNode on the heap, with correct arguments for a child node.
     */
    shared_ptr<AlphaGoDNode> AlphaGoCNode::create_child_node_helper(shared_ptr<const State> observation) const
    {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        return make_shared<AlphaGoDNode>(
            static_pointer_cast<AlphaGoManager>(thts_manager), 
            next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const AlphaGoCNode>(shared_from_this()));
    }

    /**
     * Make create_child functions make a AlphaGo child rather than a UctCNode or ThtsCNode
     */
    shared_ptr<ThtsDNode> AlphaGoCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<AlphaGoDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}