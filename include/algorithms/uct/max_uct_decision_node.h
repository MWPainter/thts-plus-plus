#pragma once

#include "algorithms/uct/max_uct_chance_node.h"
#include "algorithms/uct/uct_manager.h"
#include "algorithms/uct/uct_chance_node.h"
#include "algorithms/uct/uct_decision_node.h"
#include "thts_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {

    // forward declare corresponding UctCNode class
    class MaxUctCNode;

    /**
     * Implementation of MaxUCT - should really use DPDNode like BTS etc in future, but just kinda hacking this in
     */
    class MaxUctDNode : public UctDNode {
        friend MaxUctCNode;


        /**
         * Core ThtsDNode implementation functions.
         */
        public: 
            /**
             * Constructor
             */
            MaxUctDNode(
                std::shared_ptr<UctManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MaxUctCNode> parent=nullptr); 

            /**
             * Override fill q values to use for search - to include heuristic value
             */
            virtual void fill_q_values(std::unordered_map<std::shared_ptr<const Action>,double>& q_values) const override;
            
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
                ThtsContext& ctx);

        protected:
            /**
             * A helper function that makes a child node object on the heap and returns it. 
             * 
             * The 'create_child_node' boilerplate function uses this function to make a new child, add it to the 
             * children map. The function is marked 
             * const to enforce that we don't accidently try to duplicate logic surrounding graph search.
             * 
             * Args:
             *      action: An action to create a child node for
             * 
             * Returns:
             *      A pointer to a new UctCNode object
             */
            std::shared_ptr<MaxUctCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;
        


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
            virtual ~MaxUctDNode() = default;

            /**
             * Creates a child node, handles the internal management of the creation and returns a pointer to it.
             * 
             * This funciton is a wrapper for the create_child_node_itfc function definted in thts_decision_node.cpp, 
             * and handles the casting required to use it.
             * 
             * - If the child already exists in children, it returns a pointer to that child.
             * - If the child hasn't been created before, it makes the child (using 'create_child_node_helper'), and 
             *      inserts it appropriately into children.
             * 
             * Args:
             *      action: An action to create a child node for
             * 
             * Returns:
             *      A pointer to a new child chance node
             */
            std::shared_ptr<MaxUctCNode> create_child_node(std::shared_ptr<const Action> action);

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
            std::shared_ptr<MaxUctCNode> get_child_node(std::shared_ptr<const Action> action) const;




        /**
         * ThtsDNode interface function definitions, used by thts subroutines to interact with this node. Copied from 
         * thts_decision_node.h. 
         * 
         * Boilerplate definitions are provided in thts_decision_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsContext& ctx) override;

            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(std::shared_ptr<const Action> action) const override;
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
