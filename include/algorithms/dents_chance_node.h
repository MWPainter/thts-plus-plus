#pragma once

#include "algorithms/dents_decision_node.h"
#include "algorithms/dents_manager.h"
#include "thts_types.h"

#include "algorithms/ments_chance_node.h"
#include "algorithms/ments_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare corresponding DentsDNode class
    class DentsDNode;
    
    /**
     * An implementation of DENTS in the Thts schema
     * 
     * The implementation only really needs to change the get_temp function over MENTS.
     */
    class DentsCNode : public MentsCNode {
        // Allow DentsDNode access to private members
        friend DentsDNode;

        protected:

        public: 
            /**
             * Constructor
             */
            DentsCNode(
                std::shared_ptr<DentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const DentsDNode> parent=nullptr);

            virtual ~DentsCNode() = default;

        protected:
            /**
             * Helper to make a DentsDNode child object.
             */
            std::shared_ptr<DentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;
            
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsCNode create_child_node_itfc function can 
             * create the correct DentsDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}