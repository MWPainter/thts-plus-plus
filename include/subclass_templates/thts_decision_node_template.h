/**
 * Template for ThtsDNode subclasses, because it involves some boilerplate code that will generally look the same.
 * 
 * To use the template, copy the relevant sections into your .h and .cpp files, and make the following find and replace
 * operations:
 *      _DNode -> YourDNodeClass
 *      _CNode -> YourCNodeClass
 *      _Manager -> YourThtsManagerClass (often ThtsManager should be sufficient)
 *      _Context -> YourThtsEnvContextClass (often ThtsEnvContext should be sufficient)
 *      _Env -> YourEnvClass
 *      _S -> YourStateClass
 *      _A -> YourActionClass
 *      _O -> YourObservationClass
 * 
 * Finally, complete all of the TODO comments inline.
 * 
 * Note that much of the boilerplate code can be deleted if you don't actually use it. I think the only strictly 
 * necessary functions are 'create_child_node_helper' and 'create_child_node_helper_itfc'. See Puct implementation for 
 * an example of being able to delete much of the code and rely on subclass code.
 */

/**
 * -----------------------------------
 * .h template - copy into .h file
 * -----------------------------------
 */

#pragma once

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
    // TODO: delete these forward declarations (added to stop IDEs showing compile errors).
    class _S;
    class _A;
    class _O;
    class _Manager;
    class _Context;
    class _Env;

    // forward declare corresponding _CNode class
    class _CNode;

    /**
     * TODO: Your docstring here
     */
    class _DNode : public ThtsDNode {
        // Allow _CNode access to private members
        friend _CNode;

        /**
         * Core _DNode implementation.
         */
        protected:
            /**
             * TODO: add your member variables here
             * TODO: add any additional member functions here 
             * (Change access modifiers as needed)
             */



        /**
         * Core ThtsDNode implementation functions.
         */
        public:  
            /**
             * Constructor
             */
            _DNode(
                std::shared_ptr<_Manager> thts_manager,
                std::shared_ptr<const _S> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const _CNode> parent=nullptr); 
            
            /**
             * Implements the thts visit function for the node
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             */
            void visit(_Context& ctx);
            
            /**
             * Implements the thts select_action function for the node
             * 
             * Args:
             *      ctx: A context provided to all thts functions throughout a trial to pass intermediate/transient info
             * 
             * Returns:
             *      The selected action
             */
            std::shared_ptr<const _A> select_action(_Context& ctx);
            
            /**
             * Implements the thts recommend_action function for the node
             * 
             * Args:
             *      ctx: A context for if a recommendation also requires a context
             * 
             * Returns:
             *      The recommended action
             */
            std::shared_ptr<const _A> recommend_action(_Context& ctx) const;
            
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
                _Context& ctx);

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
             *      A pointer to a new _CNode object
             */
            std::shared_ptr<_CNode> create_child_node_helper(std::shared_ptr<const _A> action) const;

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
            virtual ~_DNode() = default;

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
            std::shared_ptr<_CNode> create_child_node(std::shared_ptr<const _A> action);

            /**
             * If this node has a child object corresponding to 'action'.
             * 
             * Args:
             *      action: An action to check if we have a child for
             * 
             * Returns:
             *      true if we have a child corresponding to 'action'
             */
            bool has_child_node(std::shared_ptr<const _A> action) const;

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
            std::shared_ptr<_CNode> get_child_node(std::shared_ptr<const _A> action) const;



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





/**
 * -----------------------------------
 * .cpp template - copy into .cpp file
 * -----------------------------------
 */

// TODO: add include for your header file

using namespace std; 

/**
 * TODO: implement your class here.
 */
namespace thts {
    _DNode::_DNode(
        shared_ptr<_Manager> thts_manager,
        shared_ptr<const _S> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const _CNode> parent) :
            ThtsDNode(
                static_pointer_cast<ThtsManager>(thts_manager),
                static_pointer_cast<const State>(state),
                decision_depth,
                decision_timestep,
                static_pointer_cast<const ThtsCNode>(parent)) {}
    
    void _DNode::visit(_Context& ctx) {
        num_visits += 1;
    }

    shared_ptr<const _A> _DNode::select_action(_Context& ctx) {
        return nullptr;
    }

    shared_ptr<const _A> _DNode::recommend_action(_Context& ctx) const {
        return nullptr;
    }

    void _DNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        _Context& ctx) 
    {
    }

    shared_ptr<_CNode> _DNode::create_child_node_helper(shared_ptr<const _A> action) const {
        return make_shared<_CNode>(
            thts_manager, 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const _DNode>(shared_from_this()));
    }

    string _DNode::get_pretty_print_val() const {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<_CNode> _DNode::create_child_node(shared_ptr<const _A> action) {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<_CNode>(new_child);
    }

    bool _DNode::has_child_node(shared_ptr<const _A> action) const {
        return ThtsDNode::has_child_node_itfc(static_pointer_cast<const Action>(action));
    }

    shared_ptr<_CNode> _DNode::get_child_node(shared_ptr<const _A> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<_CNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void _DNode::visit_itfc(ThtsEnvContext& ctx) {
        _Context& ctx_itfc = (_Context&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> _DNode::select_action_itfc(ThtsEnvContext& ctx) {
        _Context& ctx_itfc = (_Context&) ctx;
        shared_ptr<const _A> action = select_action(ctx_itfc);
        return static_pointer_cast<const Action>(action);
    }

    shared_ptr<const Action> _DNode::recommend_action_itfc(ThtsEnvContext& ctx) const {
        _Context& ctx_itfc = (_Context&) ctx;
        shared_ptr<const _A> action = recommend_action(ctx_itfc);
        return static_pointer_cast<const Action>(action);
    }

    void _DNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        _Context& ctx_itfc = (_Context&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<ThtsCNode> _DNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const _A> act_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<_CNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}