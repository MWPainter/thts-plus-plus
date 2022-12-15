#pragma once

#include "algorithms/dbdents_decision_node.h"
#include "algorithms/dents_manager.h"
#include "thts_types.h"

#include "algorithms/dbments_chance_node.h"
#include "algorithms/dbments_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare corresponding DBDentsDNode class
    class DBDentsDNode;
    
    /**
     * An implementation of DB-DENTS in the Thts schema
     * 
     * The implementation only really needs to change the get_temp function over DB-MENTS.
     */
    class DBDentsCNode : public DBMentsCNode {
        // Allow DBDentsDNode access to private members
        friend DBDentsDNode;

        protected:

        public: 
            /**
             * Constructor
             */
            DBDentsCNode(
                std::shared_ptr<DentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const DBDentsDNode> parent=nullptr);

            virtual ~DBDentsCNode() = default;

        protected:
            /**
             * Helper to make a DBDentsDNode child object.
             */
            std::shared_ptr<DBDentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;
            
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsCNode create_child_node_itfc function can 
             * create the correct DBDentsDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}