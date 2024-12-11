#pragma once

#include "algorithms/uct/uct_chance_node.h"
#include "algorithms/uct/hmcts_decision_node.h"
#include "algorithms/uct/hmcts_manager.h"
#include "thts_chance_node.h"
#include "thts_decision_node.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare corresponding HmctsDNode class
    class HmctsDNode;
    
    /**
     * Implementation of Hmcts (chance nodes) in Thts schema. 
     * 
     * Member variables:
     *      total_budget:
     *          The total budget for trials at this node
     *      total_budget_on_last_visit:
     *          The value of total_budget the last time that this node was visited
     *      budget_per_child:
     *          The target budget for trials per child in this round of sequential halving
     */
    class HmctsCNode : public UctCNode {
        // Allow HmctsDNode access to private members
        friend HmctsDNode;

        /**
         * Core HmctsCNode implementation.
         */
        protected:
            int total_budget;
            int total_budget_on_last_visit;
            std::unordered_map<std::shared_ptr<const State>,int> budget_per_child;

            /**
             * Helper to check if running in seq halving mode right now
            */
            bool running_seq_halving() const;

            /**
             * Fn for parent nodes to call to update budget
            */
            void set_new_total_budget(int budget);

            /**
             * Update budgets
            */
            void visit_update_budgets();

            /**
             * Sample observation 
             */
            std::shared_ptr<const State> sample_observation_budgeted();



        /**
         * Core ThtsCNode implementation functions. Implement in .cpp and add any docstrings.
         */
        public: 
            /**
             * Constructor
             */
            HmctsCNode(
                std::shared_ptr<HmctsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const HmctsDNode> parent=nullptr);

            /**
             * Implements the thts visit function for the node
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             */
            void visit(ThtsEnvContext& ctx);
            
            /**
             * Implements the thts sample_observation function for the node
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             * 
             * Returns:
             *      The sampled observation
             */
            std::shared_ptr<const State> sample_observation(ThtsEnvContext& ctx);

        protected:
            /**
             * A helper function that makes a child node object on the heap and returns it. 
             * 
             * The 'create_child_node' boilerplate function uses this function to make a new child, add it to the 
             * children map (or bypass making the node using the transposition table if using). The function is marked 
             * const to enforce that we don't accidently try to duplicate logic surrounding adding children and 
             * interacting with the transposition table.
             * 
             * Args:
             *      observation: The observation (next state) object leading to the child node
             * 
             * Returns:
             *      A pointer to a new HmctsDNode object
             */
            std::shared_ptr<HmctsDNode> create_child_node_helper(std::shared_ptr<const State> observation) const; 



        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_chance_node_template.h
         */
        public:
            /**
             * Mark destructor as virtual.
             */
            virtual ~HmctsCNode() = default;

            /**
             * Creates a child node, handles the internal management of the creation and returns a pointer to it.
             * 
             * This funciton is a wrapper for the create_child_node_itfc function definted in thts_decision_node.cpp, 
             * and handles the casting required to use it.
             * 
             * - If the child already exists in children, it returns a pointer to that child.
             * - (If using transposition table) If the child already exists in the transposition table, but not in 
             *      children, it adds the child to children and then returns a pointer to it.
             * - If the child hasn't been created before, it makes the child (using 'create_child_node_helper'), and 
             *      inserts it appropriately into children (and the transposition table if relevant).
             * 
             * Args:
             *      observation: The observation (next state) object leading to the child node
             * 
             * Returns:
             *      A pointer to a new child chance node
             */
            std::shared_ptr<HmctsDNode> create_child_node(std::shared_ptr<const State> observation);

            /**
             * Retrieves a child node from the children map.
             * 
             * If a child doesn't exist for the observation, an exception will be thrown.
             * 
             * Args:
             *      observation: The observation to get the corresponding child of
             * 
             * Returns:
             *      A pointer to the child node corresponding to 'observation'
             */
            std::shared_ptr<HmctsDNode> get_child_node(std::shared_ptr<const State> observation) const;



        /**
         * ThtsCNode interface function definitions, used by thts subroutines to interact with this node. Copied from 
         * thts_chance_node.h. 
         * 
         * Boilerplate definitions are provided in thts_chance_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual void visit_itfc(ThtsEnvContext& ctx);
            virtual std::shared_ptr<const Observation> sample_observation_itfc(ThtsEnvContext& ctx);
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
            // virtual std::shared_ptr<ThtsDNode> create_child_node_itfc(
            //    std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) final;
                


        /**
         * Implemented in thts_chance_node.{h,cpp}
         */
        // public:
        //     bool is_two_player_game() const;
        //     bool is_opponent() const;
        //     int get_num_children() const;

        //     bool has_child_node_itfc(std::shared_ptr<const Observation> observation) const;
        //     std::shared_ptr<ThtsDNode> get_child_node_itfc(std::shared_ptr<const Observation> observation) const;

        //     std::string get_pretty_print_string(int depth) const;

        // private:
        //     void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs) const;
    };
}