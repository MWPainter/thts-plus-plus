#pragma once

#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/hmcts_chance_node.h"
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

    // forward declare corresponding HmctsCNode class
    class HmctsCNode;

    /**
     * Implementation of HMCTS (decision nodes) in Thts schema. 
     * 
     * Note on this implementation:
     * - its implemented into the thts schema, so doesn't really get the memory benefits that a specific implementation 
     *      would get
     * - the algorithm is presented for deterministic envs, so there isn't an obvious way to handle stochastic envs. 
     *      I think the most reasonable way to allocate budget proportional to outcomes from the transition
     *      distribution, so that's what we do here
     * - could also parameterise this on the UCT implementation to use PUCT/UCT as options, but just gunna hard back in
     *      standard UCT (for now at least)
     * 
     * Member variables:
     *      total_budget:
     *          The total budget for trials at this node
     *      total_budget_on_last_visit:
     *          The value of total_budget the last time that this node was visited
     *      seq_halving_round_budget_per_child:
     *          The total budget for trials in this round of sequential halving (per child in  'seq_halving_actions')
     *      seq_halving_actions:
     *          The current set of actions that sequential halving is considering
     */
    class HmctsDNode : public UctDNode {
        // Allow HmctsCNode access to private members
        friend HmctsCNode;

        /**
         * Core HmctsDNode implementation.
         */
        protected:
            int total_budget;
            int total_budget_on_last_visit;
            int seq_halving_round_budget_per_child;
            ActionVector seq_halving_actions;

            /**
             * Helper to check if running in seq halving mode right now
            */
            bool running_seq_halving() const;

            /**
             * Fn for parent nodes to call to update budget
            */
            void set_new_total_budget(int budget);

            /**
             * Update budgets and setup actions currently considering
            */
            void visit_update_budgets();

            /**
             * Select action with sequential halving
            */
            std::shared_ptr<const Action> select_action_sequential_halving(ThtsEnvContext& ctx);



        /**
         * Core ThtsDNode implementation functions.
         */
        public: 
            /**
             * Constructor
             */
            HmctsDNode(
                std::shared_ptr<HmctsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const HmctsCNode> parent=nullptr); 
            
            /**
             * Implements the thts visit function for the node
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             */
            void visit(ThtsEnvContext& ctx);
            
            /**
             * Implements the thts select_action function for the node
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             * 
             * Returns:
             *      The selected action
             */
            std::shared_ptr<const Action> select_action(ThtsEnvContext& ctx);

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
             *      action: An action to create a child node for
             * 
             * Returns:
             *      A pointer to a new HmctsCNode object
             */
            std::shared_ptr<HmctsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;
        


        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in thts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            /**
             * Mark destructor as virtual.
             */
            virtual ~HmctsDNode() = default;

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
             *      action: An action to create a child node for
             * 
             * Returns:
             *      A pointer to a new child chance node
             */
            std::shared_ptr<HmctsCNode> create_child_node(std::shared_ptr<const Action> action);

            /**
             * Retrieves a child node from the children map.
             * 
             * If a child doesn't exist for the action, an exception will be thrown.
             * 
             * Args:
             *      action: The action to get the corresponding child of
             * 
             * Returns:
             *      A pointer to the child node corresponding to 'action'
             */
            std::shared_ptr<HmctsCNode> get_child_node(std::shared_ptr<const Action> action) const;




        /**
         * ThtsDNode interface function definitions, used by thts subroutines to interact with this node. Copied from 
         * thts_decision_node.h. 
         * 
         * Boilerplate definitions are provided in thts_decision_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual void visit_itfc(ThtsEnvContext& ctx);
            virtual std::shared_ptr<const Action> select_action_itfc(ThtsEnvContext& ctx);
            virtual std::shared_ptr<const Action> recommend_action_itfc(ThtsEnvContext& ctx) const;
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(std::shared_ptr<const Action> action) const;
            // virtual std::shared_ptr<ThtsCNode> create_child_node_itfc(std::shared_ptr<const Action> action) final;



        /**
         * Implemented in thts_decision_node.{h,cpp}
         */
        // public:
        //     bool is_leaf() const;
        //     bool is_root_node() const;
        //     bool is_two_player_game() const;
        //     bool is_opponent() const;
        //     int get_num_children() const;

        //     bool has_child_node_itfc(std::shared_ptr<const Action> action) const;
        //     std::shared_ptr<ThtsCNode> get_child_node_itfc(std::shared_ptr<const Action> action);

        //     std::string get_pretty_print_string(int depth) const;

        //     static std::shared_ptr<ThtsDNode> load(std::string& filename);
        //     bool save(std::string& filename) const;

        // private:
        //     void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs) const;
    };
}
