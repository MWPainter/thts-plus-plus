#pragma once

#include "algorithms/ments_chance_node.h"
#include "algorithms/ments_manager.h"
#include "thts_types.h"

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
    // forward declare corresponding MentsCNode class
    class MentsCNode;

    /**
     * An implementation of MENTS in the Thts schema
     * 
     * Our implemntation matches the paper https://proceedings.neurips.cc/paper/2019/file/7ffb4e0ece07869880d51662a2234143-Paper.pdf
     * mostly, with one difference (described below) and some additional functionality (described in 
     * ments_manager.h).
     * 
     * The difference is as follows:
     *  In the MENTS paper, a factor lambda = epsilon * num_actions / log(num_visits + 1), is used to reweight 
     *  between a uniform distribution and an boltzmann (energy based) policy. This lambda is the probability of 
     *  uniformly selecting an action, rather than using the boltzmann policy. To make the epsilon parameter 
     *  independent of the number of actions, we just remove the num_actions factor, to get:
     *      lambda = epsilon / log(num_visits + 1)
     * 
     * Member variables:
     *      soft_value: The soft value at this node
     *      num_backups: The number of times this node has been backed up
     *      actions: A list of valid actions that can be used at this node
     *      policy_prior: A prior policy for this state (if we have one)
     */
    class MentsDNode : public ThtsDNode {
        // Allow MentsCNode access to private members
        friend MentsCNode;

        /**
         * Core MentsDNode implementation.
         */
        protected:
            int num_backups;
            double soft_value;
            std::shared_ptr<ActionVector> actions;
            std::shared_ptr<ActionPrior> policy_prior;

            /**
             * Returns if we have a valid 'policy_prior' to use.
             * 
             * If we have a prior over the child nodes, we may want to use that. However checking 
             * 'thts_manager->prior_fn != nullptr' isn't very readible, so we provide this function.
             * 
             * Returns:
             *      if 'policy_prior' is valid and can be used
             */
            bool has_prior() const;

            /**
             * Helper to get the temperature that should be used.
             * 
             * The primary purpose of making this a function is so that it can be overriden in subclasses.
             */
            virtual double get_temp() const;

            /**
             * Helper to get the q-value of an action. Taking into account for if we are acting as an opponent.
             * 
             * If we are acting as an opponent, the values of children are generally negated to represent that. In this
             * library it's standard to store values with respect to the first player, so if this node is acting for an
             * opponent, then values in computations should be negated to reflect the opposing objective.
             * 
             * If no child node exists, this will return the value according to the default_q_value, or according to 
             * the policy_prior if it exists (discussed in more detail in ments_decision_node.cpp and ments_manager.h).
             * 
             * Assumes that we already hold the lock for the corresponding child node if it exists.
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
             * Computes the weights for each action.
             * 
             * (This excludes any probability mass from epsilon exploration).
             * 
             * Assumes that we already hold locks for all of the children.
             * 
             * Args:
             *      action_weights: 
             *          An ActionDistr to be filled with values of the form exp(q_value/temp - C), where C is equal to
             *          max(q_value/temp)
             *      sun_action_weights:
             *          A double reference to be filled with the sum of all the weights in 'action_weights'
             *      normalisation_term:
             *          A double reference to be filled with the value of C from 'action_weights' description.
             *      context:
             *          A thts env context
             */
            virtual void compute_action_weights(
                ActionDistr& action_weights, 
                double& sum_action_weights, 
                double& normalisation_term, 
                ThtsEnvContext& context) const;

            /**
             * Computes the action distribution for each action. (Including probability mass from epsilon exploration).
             * 
             * Is thread safe, and will lock children before trying to access them.
             * 
             * Args:
             *      action_distr:
             *          An ActionDistr to be filled with a normalised probability distribution to select actions with
             *      context:
             *          A thts env context
             */
            void compute_action_distribution(
                ActionDistr& action_distr, 
                ThtsEnvContext& context) const;

            /**
             * Implements select_action for ments
             * 
             * Args:
             *      ctx: A thts env context
             * 
             * Returns:
             *      The action selected.
             */
            std::shared_ptr<const Action> select_action_ments(ThtsEnvContext& ctx);

            /**
             * Implements recommend_action for ments.
             * 
             * Selects the maximum soft value from nodes. If the value of 'recommend_visit_threshold' is positive, then 
             * we choose only from nodes that have been visited at least 'recommend_visit_threshold' number of times,
             * unless no nodes have been visited that many times.
             * 
             * Returns:
             *      The recommended action.
             */
            std::shared_ptr<const Action> recommend_action_best_soft_value() const;

            /**
             * Implements a soft backup for ments.
             * 
             * Is thread safe, and will lock children before trying to access them.
             * 
             * I.e. V(s) = temp * log(sum(exp(Q(s,a)/temp)))
             * 
             * Args:
             *      ctx: A thts env context
             */
            void backup_soft(ThtsEnvContext& ctx);





        /**
         * Core ThtsDNode implementation functions.
         */
        public:  
            /**
             * Constructor
             */
            MentsDNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MentsCNode> parent=nullptr); 
            
            /**
             * Implements the thts visit function for the node
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             */
            virtual void visit(ThtsEnvContext& ctx);
            
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
            virtual std::shared_ptr<const Action> recommend_action(ThtsEnvContext& ctx) const;
            
            /**
             * Implements the thts backup function for the node
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
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

            /**
             * Returns if the node is a sink node in the environment.
             * 
             * Used to decide if this node is (and always will be) a leaf of the tree (it has no possible nodes that 
             * can be expanded).
             * 
             * Returns:
             *      If this node corresponds to a 'sink state' in the environment
             */
            virtual bool is_sink() const;

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
             *      A pointer to a new MentsCNode object
             */
            std::shared_ptr<MentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Returns a string representation of the value of this node currently. Used for pretty printing.
             * 
             * Returns:
             *      A string representing the value of this node
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
            virtual ~MentsDNode() = default;

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
            std::shared_ptr<MentsCNode> create_child_node(std::shared_ptr<const Action> action);

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
            std::shared_ptr<MentsCNode> get_child_node(std::shared_ptr<const Action> action) const;



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

            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
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
        //     std::shared_ptr<ThtsCNode> get_child_node_itfc(std::shared_ptr<const Action> action) const;

        //     std::string get_pretty_print_string(int depth) const;

        //     static std::shared_ptr<ThtsDNode> load(std::string& filename);
        //     bool save(std::string& filename) const;

        // private:
        //     void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs) const;
    };
}
