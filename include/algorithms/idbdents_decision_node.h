#pragma once

#include "algorithms/idbdents_chance_node.h"
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
    class IDBDentsCNode;

    /**
     * An implementation of I(mproved)DB-DENTS in the Thts schema
     * 
     * This implementation adds entropy backups to DB-MENTS to compute a soft value using a decayed entropy. Decayed 
     * temperature is used as a coefficient of entropy when computing soft values, search temperature is used in the 
     * energy based policy.
     * 
     * Attributes:
     *      local_entropy: The entropy of the policy local to this node
     *      subtree_entropy: The entropy of the policy over the subtree, rooted at this node
     */
    class IDBDentsDNode : public DBMentsDNode {
        // Allow IDBMentsDNode access to private members
        friend IDBDentsCNode;

        protected: 
            double local_entropy;
            double subtree_entropy;

        public:  
            /**
             * Constructor
             */
            IDBDentsDNode(
                std::shared_ptr<IDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const IDBDentsCNode> parent=nullptr); 

            virtual ~IDBDentsDNode() = default;

            /**
             * Helper to get the temperature that should be used in ments functions. (In IDENTS this will be the 
             * search temperature)
             */
            virtual double get_temp() const;            
            

            /**
             * Helper to get the temperature that should be used in ments functions. (In DENTS this will be a decayed 
             * temperature)
             */
            virtual double get_decayed_temp() const;            
            
            /**
             * Helper to get the soft q-value of an action. Taking into account for if we are acting as an opponent.
             * 
             * This computes the value of V + alpha_decayed * H, for each child, where V is the dp value of the child,
             * H is the subtree entropy of the child and alpha_decayed is the decayed temperature.
             * 
             * Args:
             *      action: 
             *          The action to get the corresponding q value for
             *      opponent_coeff: 
             *          A value of -1.0 or 1.0 for if we are acting as the opponent in a two player game or not 
             *          respectively
             */
            virtual double get_soft_q_value(std::shared_ptr<const Action> action, double opponent_coeff) const;

            /**
             * Computes the local and subtree entropy as a backup
             * 
             * Args:
             *      ctx: unused
            */
            void backup_entropy(ThtsEnvContext& ctx);
            
            /**
             * Calls both the entropy backup and dp backup from DPDNode
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

        protected:
            /**
             * Helper to make a IDBDentsCNode child object.
             */
            std::shared_ptr<IDBDentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Override the pretty print to print out the soft value, dp value, entropy and temperature
             */
            virtual std::string get_pretty_print_val() const;
        
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsDNode create_child_node_itfc function can 
             * create the correct IDBDentsCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}