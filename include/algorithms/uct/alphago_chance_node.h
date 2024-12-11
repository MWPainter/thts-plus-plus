#pragma once

#include "algorithms/uct/puct_chance_node.h"
#include "algorithms/uct/alphago_decision_node.h"
#include "algorithms/uct/alphago_manager.h"

#include <memory>

namespace thts {
    // forward declare corresponding AlphaGoDNode class
    class AlphaGoDNode;
    
    /**
     * Implementation of Alpha Go's search (chance nodes) in Thts schema. 
     * 
     * The only thing that a puct chance node needs to differently to uct chance nodes is when it makes children to 
     * make a AlphaGoDNode rather than a UctDNode
     */
    class AlphaGoCNode : public PuctCNode {
        friend AlphaGoDNode;

        public:
            AlphaGoCNode(
                std::shared_ptr<AlphaGoManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const AlphaGoDNode> parent=nullptr);

            virtual ~AlphaGoCNode() = default;

        protected:
            /**
             * Helper to make a AlphaGoDNode child object.
             */
            std::shared_ptr<AlphaGoDNode> create_child_node_helper(std::shared_ptr<const State> observation) const; 

        public:
            /**
             * Override create_child_node_helper_itfc, so that the ThtsCNode create_child_node_itfc function can 
             * create the correct AlphaGoDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}