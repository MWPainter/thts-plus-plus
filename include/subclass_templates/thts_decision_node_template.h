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
 */

/**
 * -----------------------------------
 * .h template - copy into .h file
 * -----------------------------------
 */

#pragma once

#include "helper.h"
#include "thts_chance_node.h"
#include "thts_decision_node.h"
#include "thts_env.h"
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
         * TODO: Core ThtsDNode implementation functions. Implement in .cpp and add any docstrings.
         */
        public: 
            _DNode(
                std::shared_ptr<_Manager> thts_manager,
                std::shared_ptr<_Env> thts_env,
                std::shared_ptr<const _S> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<_CNode> parent=nullptr); 
            
            void visit(_Context& ctx);
            std::shared_ptr<const _A> select_action(_Context& ctx);
            std::shared_ptr<const _A> recommend_action(_Context& ctx);
            void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                _Context& ctx);

            bool is_leaf();

        protected:
            // TODO: delete docstring + write own
            // Just creates an instance of _CNode on the heap and returns a pointer to it.
            std::shared_ptr<_CNode> create_child_node_helper(std::shared_ptr<const _A> action);

            virtual std::string get_pretty_print_val();
        
        /**
         * Boilerplate function definitions. Boilerplate implementations provided in thts_decision_node_template.h
         */
        public:
            virtual ~_DNode() = default;

            // Creates a child node, possibly grabbing it from a transposition table
            virtual std::shared_ptr<_CNode> create_child_node(std::shared_ptr<const _A> action);

            bool has_child(std::shared_ptr<const _A> action);
            std::shared_ptr<_CNode> get_child_node(std::shared_ptr<const _A> action);

        /**
         * ThtsDNode interface function definitions. Boilerplate implementations provided from 
         * thts_decision_node_template.h
         */
        public:
            virtual void visit_itfc(ThtsEnvContext& ctx);
            virtual std::shared_ptr<const Action> select_action_itfc(ThtsEnvContext& ctx);
            virtual std::shared_ptr<const Action> recommend_action_itfc(ThtsEnvContext& ctx);
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx);

            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(std::shared_ptr<const Action> action);
            // virtual std::shared_ptr<ThtsCNode> create_child_node_itfc(std::shared_ptr<const Action> action) final;

        /**
         * Implemented in thts_decision_node.cpp
         */
        // public:
        //     bool is_root_node();
        //     bool is_two_player_game();
        //     bool is_opponent();
        //     int get_num_children();

        //     bool has_child_itfc(std::shared_ptr<const Action> action);
        //     std::shared_ptr<ThtsCNode> get_child_node_itfc(std::shared_ptr<const Action> action);

        //     std::string get_pretty_print_string(int depth);

        //     static std::shared_ptr<ThtsDNode> load(std::string& filename);
        //     bool save(std::string& filename);

        // private:
        //     void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs);
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
        shared_ptr<_Env> thts_env,
        shared_ptr<const _S> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<_CNode> parent) :
            ThtsDNode(
                static_pointer_cast<ThtsManager>(thts_manager),
                static_pointer_cast<ThtsEnv>(thts_env),
                static_pointer_cast<const State>(state),
                decision_depth,
                decision_timestep,
                static_pointer_cast<ThtsCNode>(parent)) {}
    
    void _DNode::visit(_Context& ctx) {
        num_visits += 1;
    }

    shared_ptr<const _A> _DNode::select_action(_Context& ctx) {
        return nullptr;
    }

    shared_ptr<const _A> _DNode::recommend_action(_Context& ctx) {
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

    bool is_leaf() {
        return true;
    }

    shared_ptr<_CNode> _DNode::create_child_node_helper(shared_ptr<const _A> action) {
        return make_shared<_CNode>(
            thts_manager, 
            thts_env, 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            this);
    }

    string _DNode::get_pretty_print_val() {
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

    bool _DNode::has_child(shared_ptr<const _A> action) {
        return ThtsDNode::has_child_itfc(static_pointer_cast<const Action>(action));
    }
    shared_ptr<_CNode> _DNode::get_child_node(shared_ptr<const _A> action) {
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

    shared_ptr<const Action> _DNode::recommend_action_itfc(ThtsEnvContext& ctx) {
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

    shared_ptr<ThtsCNode> _DNode::create_child_node_helper_itfc(shared_ptr<const Action> action){
        shared_ptr<const _A> act_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<_CNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}