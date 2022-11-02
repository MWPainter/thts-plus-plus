#pragma once

#include "helper.h"
#include "thts_decision_node.h"
#include "thts_env.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare
    class ThtsDNode;

    // CNodeMap type is lengthy, so typedef
    typedef std::unordered_map<std::shared_ptr<const Observation>,std::shared_ptr<ThtsDNode>> DNodeChildMap;
    
    /**
     * TODO: abstract base class for CNodes, list of attributes.
     * TODO: write around parent = the parent that CONSTRUCTED this node
     * TODO: create a superclass ThtsNode, that shoves all of the shared functionality of the decision and chance nodes together
     */
    class ThtsCNode {
        // Allow ThtsDNode access to private members
        friend ThtsDNode;

        protected:
            std::shared_ptr<ThtsManager> thts_manager;
            std::shared_ptr<ThtsEnv> thts_env;
            std::shared_ptr<const State> state;
            std::shared_ptr<const Action> action;
            int decision_depth;
            int decision_timestep;

            int num_visits;
            std::shared_ptr<ThtsDNode> parent;
            DNodeChildMap children;

            HeuristicFnPtr heuristic_fn_ptr;

        public: 
            /**
             * Default constructor.
             * 
             * Initialises the attributes of the class.
             */
            ThtsCNode(
                std::shared_ptr<ThtsManager> thts_manager,
                std::shared_ptr<ThtsEnv> thts_env,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<ThtsDNode> parent=nullptr,
                HeuristicFnPtr heuristic_fn_ptr=&helper::zero_heuristic_fn);

            /**
             * Default destructor is sufficient. But need to declare it virtual.
             */
            virtual ~ThtsCNode() = default;

            /**
             * Thts visit function.
             * 
             * Called everytime the thts routine selects this node.
             * 
             * Args:
             *      ctx: object holding context information for the current trial   
             */
            virtual void visit_itfc(ThtsEnvContext& ctx);

            /**
             * The sample observation function. Selects an observation to explore from this node.
             * 
             * Used int he selection phase of the thts routine to select observations at this node.
             * 
             * Args:
             *      ctx: object holding context information for the current trial 
             *      
             * Returns:
             *      The sampled observation
             */
            virtual std::shared_ptr<Observation> sample_observation_itfc(ThtsEnvContext& ctx) = 0;

            /**
             * Thts backup function.
             * 
             * Updates the information in this node in the backup phase of the thts routine.
             * 
             * Args:
             *      trial_rewards_before_node: 
             *          A list of rewards recieved (at each timestep) on the trial prior to reaching this node.
             *      trial_rewards_after_node:
             *          A list of rewards recieved (at each timestep) on the trial after reaching this node. This list 
             *          includes the reward from R(state,action) that would have been recieved from taking the action 
             *          in this node.
             *      trial_cumulative_return_after_node:
             *          Sum of rewards in the 'trial_rewards_after_node' list
             *      trial_cumulative_return:
             *          Sum of rewards in both of the 'trial_rewards_after_node' and 'trial_rewards_before_node' lists
             */
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx) = 0;

            /**
             * Creates a child node and inserts it in the unordered_map 'children'.
             * 
             * This virtual final method means that this implementation cannot be overriden. This is to protect the 
             * logic surrounding the transposition table, which is found in the 'thts_manager' object. It will perform 
             * the following logic:
             *      - if not using transposition table:
             *          - make child node using 'create_child_node_helper' and insert in children map
             *      - if using transposition table:
             *          - check transposition table for child node, if it exists, adds to children map and returns
             *          - otherwise creates the child node, and inserts it into the children map and transposition table
             * 
             * Args:
             *      observation: The observation to create a child node for 
             * 
             * Returns:
             *      A pointer to the created child node
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_itfc(std::shared_ptr<const Observation> observation) final;

        protected:
            /**
             * Helper for constructing a child node. Should create and instance and return a pointer to it.
             * 
             * Args:
             *      observation: The observation to create a child node for 
             * 
             * Returns:
             *      A pointer to a newly created child node on the heap
             */
            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(std::shared_ptr<const Observation> observation) = 0;

            /**
             * Computes the next state given the current state (stored in this object) and an observation. 
             * 
             * For fully observable cases, this should just involve casting the observation object
             * 
             * Args:
             *      observation: The observation we want to use to compute a successor state
             * 
             * Returns:
             *      A successor state (corresponding to the state for children[observation])
             */
            virtual std::shared_ptr<const State> compute_next_state_from_observation_itfc(
                std::shared_ptr<const Observation> observation) = 0;

            /**
             * Helper for pretty printing. Should return some string representing the current 'value' of this node.
             * 
             * Returns:
             *      string representing the current value of this node
             */
            virtual std::string get_pretty_print_val() = 0;

        public:

            /**
             * Returns if this node is planning for a two player game.
             * 
             * Returns:
             *      True if currently planning for a two player game
             */
            bool is_two_player_game();

            /**
             * Returns if this node is planning as the opponent in a two player game.
             * 
             * If not a two player game, this will always return false.
             * 
             * Returns:
             *      If this node is planning as the opponent in a two player game
             */
            bool is_opponent();

            /**
             * Helper function to get number of children this node currently has.
             * 
             * Returns:
             *      Number of children in 'children' map
             */
            int get_num_children();

            /**
             * Helper function to check if node has a child for the given observation
             * 
             * Args:
             *      observation: The observation to check if we have a child for
             * 
             * Returns:
             *      Returns true if node has a child corresponding to 'observation'
             */
            bool has_child_itfc(std::shared_ptr<const Observation> observation);

            /**
             * Returns a pointer to a child of this node.
             * 
             * Args:
             *      observation: The observation that we want the corresponding child node for.
             * 
             * Returns:
             *      A pointer to the child decision node.
             */
            std::shared_ptr<ThtsDNode> get_child_node_itfc(std::shared_ptr<const Observation> observation);

            /**
             * Pretty prints the tree to a string.
             * 
             * Args:
             *      depth: To what (decision) depth do we want to print to?
             * 
             * Returns:
             *      A string that is a pretty representation of the top part of the tree, rooted at this node
             */
            std::string get_pretty_print_string(int depth);

        private:
            /**
             * TODO: write docstring
             */
            void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs);
    };
}