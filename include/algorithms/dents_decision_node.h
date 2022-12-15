#pragma once

#include "algorithms/dents_chance_node.h"
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
    // forward declare corresponding DentsCNode class
    class DentsCNode;

    /**
     * An implementation of DENTS in the Thts schema
     * 
     * The implementation only really needs to change the get_temp function over MENTS.
     */
    class DentsDNode : public MentsDNode {
        // Allow DentsCNode access to private members
        friend DentsCNode;

        protected: 

        public:  
            /**
             * Constructor
             */
            DentsDNode(
                std::shared_ptr<DentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const DentsCNode> parent=nullptr); 

            virtual ~DentsDNode() = default;

            /**
             * Helper to get the temperature that should be used.
             */
            virtual double get_temp() const;

        protected:
            /**
             * Helper to make a DentsCNode child object.
             */
            std::shared_ptr<DentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Override the pretty print to print out the soft value and the temp for debugging
             */
            virtual std::string get_pretty_print_val() const;
        
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the ThtsDNode create_child_node_itfc function can 
             * create the correct DentsCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}
