#pragma once

#include "algorithms/uct_decision_node.h"
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
    // forward declare corresponding UctDNode class
    class UctDNode;
    
    /**
     * Implementation of UCT (decision nodes) in Thts schema. 
     * 
     * Member variables:
     *      next_state_distr: A cached StateDistribution, representing the distribution over possible next states
     *      avg_return: The average return from this node
     */
    class UctCNode : public ThtsCNode {
        // Allow UctDNode access to private members
        friend UctDNode;

        /**
         * Core UctCNode implementation.
         */
        protected:
            std::shared_ptr<StateDistr> next_state_distr;
            double avg_return;

            /**
             * Handles the thts sample_observation function by randomly sampling.
             * 
             * Returns:
             *      The sampled next state
             */
            std::shared_ptr<const State> sample_observation_random();

            /**
             * Performs an 'avg_return' backup, i.e. it encorporates a new value into the current average.
             * 
             * Args:
             *      trial_return_after_node: The cumulative return achieved after this node, for the current trial
             */
            void backup_average_return(const double trial_return_after_node);



        /**
         * Core ThtsCNode implementation functions. Implement in .cpp and add any docstrings.
         */
        public: 
            /**
             * Constructor
             */
            UctCNode(
                std::shared_ptr<UctManager> thts_manager,
                std::shared_ptr<ThtsEnv> thts_env,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const UctDNode> parent=nullptr);

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
            void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

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
             *      observation: An observation to create a child node for
             * 
             * Returns:
             *      A pointer to a new _DNode object
             */
            std::shared_ptr<UctDNode> create_child_node_helper(std::shared_ptr<const State> observation) const;

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
         * Boilerplate implementations provided in thts_chance_node_template.h
         */
        public:
            /**
             * Mark destructor as virtual.
             */
            virtual ~UctCNode() = default;

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
            std::shared_ptr<UctDNode> create_child_node(std::shared_ptr<const State> observation);

            /**
             * If this node has a child object corresponding to 'observation'.
             * 
             * Args:
             *      observation: An observation to check if we have a child for
             * 
             * Returns:
             *      true if we have a child corresponding to 'observation'
             */
            bool has_child_node(std::shared_ptr<const State> observation) const;

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
            std::shared_ptr<UctDNode> get_child_node(std::shared_ptr<const State> observation) const;



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

            virtual std::shared_ptr<const State> compute_next_state_from_observation_itfc(
                std::shared_ptr<const Observation> observation) const;

            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation) const;
            // virtual std::shared_ptr<ThtsDNode> create_child_node_itfc(
            //    std::shared_ptr<const Observation> observation) final;
                


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