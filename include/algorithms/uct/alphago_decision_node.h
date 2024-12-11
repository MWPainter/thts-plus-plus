#pragma once

#include "algorithms/uct/puct_decision_node.h"
#include "algorithms/uct/alphago_chance_node.h"
#include "algorithms/uct/alphago_manager.h"

#include <memory>

namespace thts {
    // forward declare corresponding AlphaGoCNode class
    class AlphaGoCNode;

    /**
     * Implementation of Alpha Go's search (decision nodes) in Thts schema. 
     * 
     * Just adds dirichlet noise at the root 
     * 
     * Member variables:
     *      base_policy_prior: The base value of the 'policy_prior' variable, for noise to be added to
     */
    class AlphaGoDNode : public PuctDNode {
        friend AlphaGoCNode;

        protected:
            std::shared_ptr<ActionPrior> base_policy_prior;

        public: 
            /**
             * Constructor
             */
            AlphaGoDNode(
                std::shared_ptr<AlphaGoManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const AlphaGoCNode> parent=nullptr); 

            virtual ~AlphaGoDNode() = default;

        protected:
            /**
             * Add dirichlet noise to policy_prior
             */
            void add_dirichlet_noise_to_prior();
            
            /**
             * Implements the thts visit function for the node
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             */
            void visit(ThtsEnvContext& ctx);

            /**
             * Helper to make a PuctCNode child object.
             */
            std::shared_ptr<AlphaGoCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

        public:
            /**
             * Override create_child_node_helper_itfc, so that the ThtsDNode create_child_node_itfc function can 
             * create the correct PuctCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(std::shared_ptr<const Action> action) const;

            /**
             * Override visit itfc function to point it at our visit function
            */
            virtual void visit_itfc(ThtsEnvContext& ctx);
    };
}