#pragma once

#include "algorithms/idbdents_decision_node.h"
#include "algorithms/idents_manager.h"
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
    // forward declare corresponding IDBDentsDNode class
    class IDBDentsDNode;
    
    /**
     * An implementation of I(mproved)DB-DENTS in the Thts schema
     * 
     * The implementation only needs to change backup functions to use dp backups too.
     * 
     * Attributes:
     *      subtree_entropy: The entropy of the policy over the subtree, rooted at this node
     */
    class IDBDentsCNode : public DBMentsCNode {
        // Allow IDBDentsDNode access to private members
        friend IDBDentsDNode;

        protected:
            double subtree_entropy;

        public: 
            /**
             * Constructor
             */
            IDBDentsCNode(
                std::shared_ptr<IDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const IDBDentsDNode> parent=nullptr);

            virtual ~IDBDentsCNode() = default;
            
            /**
             * Computes the local and subtree entropy as a backup
             * 
             * Args:
             *      ctx: unused
            */
            void backup_entropy();

            /**
             * Calls both the entropy backup and dp backup from DPCNode
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

        protected:
            /**
             * Helper to make a IDBDentsDNode child object.
             */
            std::shared_ptr<IDBDentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;

            /**
             * Override the pretty print to print out both the dp value and the soft value
             */
            virtual std::string get_pretty_print_val() const;
            
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsCNode create_child_node_itfc function can 
             * create the correct IDBDentsDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}