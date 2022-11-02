#pragma once

#include "helper.h"
#include "thts_chance_node.h"
#include "thts_env.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    // forward declare
    class ThtsCNode;

    // CNodeMap type is lengthy, so typedef
    typedef std::unordered_map<std::shared_ptr<const Action>,std::shared_ptr<ThtsCNode>> CNodeChildMap;

    /**
     * TODO: abstract base class for DNodes, list of attributes.
     * TODO: write around parent = the parent that CONSTRUCTED this node
     */
    class ThtsDNode {
        // Allow ThtsCNode access to private members
        friend ThtsCNode;

        protected:
            std::shared_ptr<ThtsManager> thts_manager;
            std::shared_ptr<ThtsEnv> thts_env;
            std::shared_ptr<const State> state;
            int decision_depth;
            int decision_timestep;

            int num_visits;
            std::shared_ptr<ThtsCNode> parent;
            CNodeChildMap children;

            HeuristicFnPtr heuristic_fn_ptr;
            PriorFnPtr prior_fn_ptr;

        public: 
            /**
             * Constructor.
             * 
             * Initialises the attributes of the class.
             */
            ThtsDNode(
                std::shared_ptr<ThtsManager> thts_manager,
                std::shared_ptr<ThtsEnv> thts_env,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<ThtsCNode> parent=nullptr,
                HeuristicFnPtr heuristic_fn_ptr=&helper::zero_heuristic_fn,
                PriorFnPtr prior_fn_ptr=nullptr); 

            /**
             * Default destructor is sufficient. But need to declare it virtual.
             */
            virtual ~ThtsDNode() = default;

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
             * Thts select action function. Selects an action to explore fromt this node
             * 
             * Used in the selection phase of the thts routine to select actions at this node.
             * 
             * Args:
             *      ctx: object holding context information for the current trial 
             *      
             * Returns:
             *      The selected action
             */
            virtual std::shared_ptr<Action> select_action_itfc(ThtsEnvContext& ctx) = 0;

            /**
             * Recommends an action from this node.
             * 
             * Recommends what this node considers to be the best action to take from its current state.
             * 
             * Args:
             *      ctx: object holding context information for the current trial 
             * 
             * Returns:
             *      The recommended action
             */
            virtual std::shared_ptr<Action> recommend_action_itfc(optional<ThtsEnvContext>& ctx) = 0;

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
             *          includes the reward from R(state,action) that would have been recieved from taking an action 
             *          from this node.
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
             * Returns if the node is a leaf node (with respect to the environment, NOT the tree).
             * 
             * Use to decide if this a 'true' leaf of the tree (it has no possible nodes that can be expanded).
             * 
             * Returns:
             *      If this node corresponds to a 'leaf state' in the environment
             */
            virtual bool is_leaf() = 0;

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
             *      action: The action to create a child node for 
             * 
             * Returns:
             *      A pointer to the created child node
             */
            virtual std::shared_ptr<ThtsCNode> create_child_node_itfc(std::shared_ptr<const Action> action) final;

        protected:
            /**
             * Helper for constructing a child node. Should create and instance and return a pointer to it.
             * 
             * Args:
             *      action: The action to create a child node for 
             * 
             * Returns:
             *      A pointer to a newly created child node on the heap
             */
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(std::shared_ptr<const Action> action) = 0;

            /**
             * Helper for pretty printing. Should return some string representing the current 'value' of this node.
             * 
             * Returns:
             *      string representing the current value of this node
             */
            virtual string get_pretty_print_val() = 0;

        public:
            /**
             * Returns if this node is the root node of the tree.
             * 
             * Returns:
             *      True if current node is the root node
             */
            bool is_root_node();

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
             * Helper function to check if node has a child for the given action
             * 
             * Args:
             *      action: The action to check if we have a child for
             * 
             * Returns:
             *      Returns true if node has a child corresponding to 'action'
             */
            bool has_child_itfc(std::shared_ptr<const Action> action);

            /**
             * Returns a pointer to a child of this node.
             * 
             * Args:
             *      action: The action that we want the corresponding child node for.
             * 
             * Returns:
             *      A pointer to the child chance node.
             */
            std::shared_ptr<ThtsCNode> get_child_node_itfc(std::shared_ptr<const Action> action);

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

            /**
             * Loads a tree from a given filename.
             * 
             * Args:
             *      filename: The filename to look for a tree file at
             * 
             * Returns:
             *      A ThtsDNode which is the root node of a Thts tree
             */
            static std::shared_ptr<ThtsDNode> load(std::string& filename);

            /**
             * Saves the tree to a given filename.
             * 
             * Args:
             *      filename: The filename to save this tree as an object to
             * 
             * Returns:
             *      True if saving was successful.
             */
            bool save(std::string& filename);

        private:
            /**
             * TODO: write docstring
             */
            void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs);
    };
}