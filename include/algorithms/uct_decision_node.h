#pragma once

#include "algorithms/uct_chance_node.h"
#include "algorithms/uct_manager.h"
#include "thts_chance_node.h"
#include "thts_decision_node.h"
#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {

    // forward declare corresponding UctCNode class
    class UctCNode;

    /**
     * Implementation of UCT (decision nodes) in Thts schema. 
     * 
     * Member variables:
     *      actions: A cached list of actions
     *      avg_return: The average return from this node
     *      policy_prior: A map from actions to probabilities representing a policy prior (over action/child nodes)
     */
    class UctDNode : public ThtsDNode {
        // Allow UctCNode access to private members
        friend UctCNode;

        /**
         * Core UctDNode implementation.
         */
        protected:
            std::shared_ptr<ActionVector> actions;
            double avg_return;
            std::unordered_map<std::shared_ptr<const Action>,double> policy_prior;

            /**
             * Returns if we have a valid 'policy_prior' to use.
             * 
             * If we have a prior over the child nodes, we may want to use that. However checking 
             * 'thts_manager->prior_fn != nullptr' isn't very readible, so we provide this function.
             * 
             * Returns:
             *      if 'policy_prior' is valid and can be used
             */
            inline bool has_prior() const;

            /**
             * Computes the ucb term for a single child for use in selecting actions.
             * 
             * Args:
             *      num_visits: The number of visits to this node
             *      child_visits: The number of time the child node has been visited
             * 
             * Returns:
             *      The confidence interval term for a ucb value
             */
            virtual double compute_ucb_term(int num_visits, int child_visits) const;

            /**
             * Helper function for 'select_action_ucb' that computes the ucb values
             * 
             * Args:
             *      ucb_values: An unordered map to be filled with ucb values by this function 
             *      ctx: The thts context given to a select aciton call
             * 
             * Returns:
             *      A map from actions to their corresponding ucb values
             */
            void fill_ucb_values(
                std::unordered_map<std::shared_ptr<const Action>,double>& ucb_values, ThtsEnvContext& ctx) const;

            /**
             * Implementation of thts 'select_action' function: that selects actions according to a hybrid 
             * implementation of the ucb and pucb algorithms.
             * 
             * Args:
             *      ctx: The thts context given to a select aciton call
             * 
             * Returns:
             *      The selected action
             */
            virtual std::shared_ptr<const Action> select_action_ucb(ThtsEnvContext& ctx);

            /**
             * An implementation thts 'select_action' function: that selects a uniformly random action. 
             * 
             * Returns:
             *      The selected action
             */
            std::shared_ptr<const Action> select_action_random();

            /**
             * Recommends the action corresponding to the child with the best avg_return.
             * 
             * Returns:
             *      An action recomendation for the state corresponding to this node
             */
            std::shared_ptr<const Action> recommend_action_best_empirical() const;

            /**
             * Recommends the action corresponding to the most visited child node.
             * 
             * Returns:
             *      An action recomendation for the state corresponding to this node
             */
            std::shared_ptr<const Action> recommend_action_most_visited() const;

            /**
             * Performs an 'avg_return' backup, i.e. it encorporates a new value into the current average.
             * 
             * Args:
             *      trial_return_after_node: The cumulative return achieved after this node, for the current trial
             */
            void backup_average_return(const double trial_return_after_node);



        /**
         * Core ThtsDNode implementation functions.
         */
        public: 
            /**
             * Constructor
             */
            UctDNode(
                std::shared_ptr<UctManager> thts_manager,
                std::shared_ptr<ThtsEnv> thts_env,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const UctCNode> parent=nullptr); 
            
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
            
            /**
             * Implements the thts recommend_action function for the node
             * 
             * Args:
             *      ctx: A context for if a recommendation also requires a context
             * 
             * Returns:
             *      The recommended action
             */
            std::shared_ptr<const Action> recommend_action(ThtsEnvContext& ctx) const;
            
            /**
             * Implements the thts backup function for the node
             * 
             * Args:
             * 
             * Args:
             *      trial_rewards_before_node: 
             *          A list of rewards recieved (at each timestep) on the trial prior to reaching this node.
             *      trial_rewards_after_node:
             *          A list of rewards recieved (at each timestep) on the trial after reaching this node. This list 
             *          includes the reward from R(state,action) that would have been recieved from taking an action 
             *          from this node.
             *      trial_cumulative_return_after_node:
             *          Sum of rewards in the 'trial_rewards_after_node' list
             *      trial_cumulative_return:
             *          Sum of rewards in both of the 'trial_rewards_after_node' and 'trial_rewards_before_node' lists
             */
            void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

            /**
             * Returns if the node is a leaf node (with respect to the environment, NOT the tree).
             * 
             * Use to decide if this a 'true' leaf of the tree (it has no possible nodes that can be expanded).
             * 
             * Returns:
             *      If this node corresponds to a 'leaf state' in the environment
             */
            virtual bool is_leaf() const;

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
             *      A pointer to a new UctCNode object
             */
            std::shared_ptr<UctCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Returns a string representation of the value of this node currently. Used for pretty printing.
             * 
             * Returns:
             *      A string of 'avg_return'
             */
            virtual std::string get_pretty_print_val() const;
        


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
            virtual ~UctDNode() = default;

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
            virtual std::shared_ptr<UctCNode> create_child_node(std::shared_ptr<const Action> action);

            /**
             * If this node has a child object corresponding to 'action'.
             * 
             * Args:
             *      action: An action to check if we have a child for
             * 
             * Returns:
             *      true if we have a child corresponding to 'action'
             */
            bool has_child_node(std::shared_ptr<const Action> action) const;

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
            std::shared_ptr<UctCNode> get_child_node(std::shared_ptr<const Action> action) const;




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
