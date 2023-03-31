#pragma once

#include "algorithms/idents_decision_node.h"
#include "algorithms/idents_manager.h"
#include "thts_types.h"

#include "algorithms/idbdents_chance_node.h"
#include "algorithms/idbdents_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare corresponding IDentsDNode class
    class IDentsDNode;
    
    /**
     * An implementation of I(mproved)DB-DENTS in the Thts schema
     * 
     * This implementation doesn't really need to change anything from IDBDents
     */
    class IDentsCNode : public IDBDentsCNode {
        // Allow IDentsDNode access to private members
        friend IDentsDNode;

        protected:

        public: 
            /**
             * Constructor
             */
            IDentsCNode(
                std::shared_ptr<IDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const IDentsDNode> parent=nullptr);

            virtual ~IDentsCNode() = default;

        protected:
            /**
             * Helper to make a IDentsDNode child object.
             */
            std::shared_ptr<IDentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;

            /**
             * Override the pretty print to print out both the dp value and the soft value
             */
            virtual std::string get_pretty_print_val() const;
            
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsCNode create_child_node_itfc function can 
             * create the correct IDentsDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}