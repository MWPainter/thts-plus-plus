#pragma once

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
     * An abstract base class for Decision Node.
     * 
     * This class provides some base implementations that can be useful across different Thts algorithms. Including 
     * a transposition table implementation and pretty print functions for debugging.
     * 
     * Member variables:
     *      thts_manager: 
     *          A ThtsManager object that stores the 'global' information about how the Thts algorithm should operate,
     *          so that an implementation can provide multiple modes of operation. Additionally stores the 
     *          transposition tables
     *      thts_env:
     *          A ThtsEnv object that provides the dynamics of the environment being planned in
     *      state:
     *          The state associated with this node, which we want to make a decision for (what is the best action)
     *      decision_depth:
     *          The decision depth of the node in this tree
     *      decision_timestep:
     *          The timestep corresponding to the current state in the larger planning problem. This is necessary 
     *          when the Thts algorithm is used at each timestep to make a decision. For example, this is necessary in 
     *          a two player game to decide who's turn it is
     *      num_visits:
     *          The number of times the node has been visited (had the 'visit' function called)
     *      parent:
     *          A pointer to this nodes parent node. nullptr if this node is the root node
     *      children:
     *          A map from Action objects to child ThtsCNode objects
     */
    class ThtsDNode : public std::enable_shared_from_this<ThtsDNode> {
        // Allow ThtsCNode access to private members
        friend ThtsCNode;

        protected:
            std::shared_ptr<ThtsManager> thts_manager;
            std::shared_ptr<ThtsEnv> thts_env;
            std::shared_ptr<const State> state;
            int decision_depth;
            int decision_timestep;
            std::shared_ptr<const ThtsCNode> parent;

            int num_visits;
            CNodeChildMap children;

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
                std::shared_ptr<const ThtsCNode> parent=nullptr); 

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
            virtual std::shared_ptr<const Action> select_action_itfc(ThtsEnvContext& ctx) = 0;

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
            virtual std::shared_ptr<const Action> recommend_action_itfc(ThtsEnvContext& ctx) const = 0;

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
            virtual bool is_leaf() const;

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
            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const = 0;

            /**
             * Helper for pretty printing. Should return some string representing the current 'value' of this node.
             * 
             * Returns:
             *      string representing the current value of this node
             */
            virtual std::string get_pretty_print_val() const = 0;

        public:
            /**
             * Returns if this node is the root node of the tree.
             * 
             * Returns:
             *      True if current node is the root node
             */
            bool is_root_node() const;

            /**
             * Returns if this node is planning for a two player game.
             * 
             * Returns:
             *      True if currently planning for a two player game
             */
            bool is_two_player_game() const;

            /**
             * Returns if this node is planning as the opponent in a two player game.
             * 
             * If not a two player game, this will always return false.
             * 
             * Returns:
             *      If this node is planning as the opponent in a two player game
             */
            bool is_opponent() const;

            /**
             * Helper function to get number of children this node currently has.
             * 
             * Returns:
             *      Number of children in 'children' map
             */
            int get_num_children() const;

            /**
             * Helper function to check if node has a child for the given action
             * 
             * Args:
             *      action: The action to check if we have a child for
             * 
             * Returns:
             *      Returns true if node has a child corresponding to 'action'
             */
            bool has_child_node_itfc(std::shared_ptr<const Action> action) const;

            /**
             * Returns a pointer to a child of this node.
             * 
             * Args:
             *      action: The action that we want the corresponding child node for.
             * 
             * Returns:
             *      A pointer to the child chance node.
             */
            std::shared_ptr<ThtsCNode> get_child_node_itfc(std::shared_ptr<const Action> action) const;

            /**
             * Pretty prints the tree to a string.
             * 
             * Args:
             *      depth: To what (decision) depth do we want to print to?
             * 
             * Returns:
             *      A string that is a pretty representation of the top part of the tree, rooted at this node
             */
            std::string get_pretty_print_string(int depth) const;

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
            bool save(std::string& filename) const;

        private:
            /**
             * A helper function that actually implements 'get_pretty_pring_string' above.
             */
            void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs) const;
    };
}