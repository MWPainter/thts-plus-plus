#pragma once

#include "algorithms/rents_decision_node.h"
#include "algorithms/ments_manager.h"
#include "thts_types.h"

#include "ments_chance_node.h"
#include "ments_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare corresponding RentsDNode class
    class RentsDNode;
    
    /**
     * Implementation of Tents chance nodes in the Thts schema. (Doesn't need to do anything above MentsCNode)
     */
    class RentsCNode : public MentsCNode {
        // Allow RentsDNode access to private members
        friend RentsDNode;

        protected:



        /**
         * Core ThtsCNode implementation functions.
         */
        public: 
            /**
             * Constructor
             */
            RentsCNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const RentsDNode> parent=nullptr);

            virtual ~RentsCNode() = default;

        protected:
            /**
             * A helper function that makes a child node object on the heap and returns it. 
             * 
             * Just need to make a child rents node, rather than child ments node.
             * 
             * Args:
             *      observation: The observation object leading to the child node
             *      next_state: The next state to construct the child node with
             * 
             * Returns:
             *      A pointer to a new RentsDNode object
             */
            std::shared_ptr<RentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;



        /**
         * ThtsCNode interface function definitions, used by thts subroutines to interact with this node. Copied from 
         * thts_chance_node.h. 
         * 
         * Boilerplate definitions are provided in thts_chance_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}