#pragma once

#include "algorithms/idents_chance_node.h"
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
    class IDentsCNode;

    /**
     * An implementation of I(mproved)DENTS in the Thts schema
     * 
     * Interestingly, as I-DENTS uses a mixture of DP backup and entropy backup, it's actually simpler to implement 
     * IDBDENTS first, then just inherit from that and change the recommendation function.
     */
    class IDentsDNode : public IDBDentsDNode {
        // Allow IDBMentsDNode access to private members
        friend IDBDentsCNode;

        protected: 
            double local_entropy;
            double subtree_entropy;

        public:  
            /**
             * Constructor
             */
            IDentsDNode(
                std::shared_ptr<IDentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const IDentsCNode> parent=nullptr); 

            virtual ~IDentsDNode() = default;
            
            /**
             * To implement IDENTS, we just need to change the recommendation function to use the MENTS recommendation
             * rather than 
             */
            virtual std::shared_ptr<const Action> recommend_action(ThtsEnvContext& ctx) const;

        protected:
            /**
             * Helper to make a IDentsCNode child object.
             */
            std::shared_ptr<IDentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Override the pretty print to print out the soft value, dp value, entropy and temperature
             */
            virtual std::string get_pretty_print_val() const;
        
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsDNode create_child_node_itfc function can 
             * create the correct IDentsCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}