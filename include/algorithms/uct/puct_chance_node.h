#pragma once

#include "algorithms/uct/uct_chance_node.h"
#include "algorithms/uct/puct_decision_node.h"
#include "algorithms/uct/puct_manager.h"

#include <memory>

namespace thts {
    // forward declare corresponding PuctDNode class
    class PuctDNode;
    
    /**
     * Implementation of PUCT (chance nodes) in Thts schema. 
     * 
     * The only thing that a puct chance node needs to differently to uct chance nodes is when it makes children to 
     * make a PuctDNode rather than a UctDNode
     */
    class PuctCNode : public UctCNode {
        friend PuctDNode;

        public:
            PuctCNode(
                std::shared_ptr<PuctManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const PuctDNode> parent=nullptr);

            virtual ~PuctCNode() = default;

        protected:
            /**
             * Helper to make a PuctDNode child object.
             */
            std::shared_ptr<PuctDNode> create_child_node_helper(std::shared_ptr<const State> observation) const; 

        public:
            /**
             * Override create_child_node_helper_itfc, so that the ThtsCNode create_child_node_itfc function can 
             * create the correct PuctDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}