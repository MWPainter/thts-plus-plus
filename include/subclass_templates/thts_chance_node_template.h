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

    // forward declare corresponding _DNode class
    class _DNode;
    
    /**
     * TODO: abstract base class for CNodes, list of attributes.
     * TODO: write around parent = the parent that CONSTRUCTED this node
     * TODO: create a superclass ThtsNode, that shoves all of the shared functionality of the decision and chance nodes together
     */
    class _CNode : public ThtsCNode {
        // Allow ThtsDNode access to private members
        friend _DNode;

        /**
         * TODO: Core DNode implementation functions. Implement in .cpp and add any docstrings.
         */
        public: 
            _CNode(
                std::shared_ptr<_Manager> thts_manager,
                std::shared_ptr<_Env> thts_env,
                std::shared_ptr<const _S> state,
                std::shared_ptr<const _A> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<ThtsDNode> parent=nullptr);

            void visit(_Context& ctx);
            std::shared_ptr<const _O> sample_observation(_Context& ctx);
            void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                _Context& ctx);

            std::shared_ptr<const _S> compute_next_state_from_observation(std::shared_ptr<const _O> observation);

        protected:
            // TODO: delete docstring + write own
            // Just creates an instance of _DNode on the heap and returns a pointer to it.
            std::shared_ptr<_DNode> create_child_node_helper(std::shared_ptr<const _O> observation);

            virtual std::string get_pretty_print_val();

        /**
         * Boilerplate function definitions. Boilerplate implementations provided in thts_chance_node_template.h
         */
        public:
            virtual ~_CNode() = default;

            // Creates a child node, possibly grabbing it from a transposition table
            std::shared_ptr<_DNode> create_child_node_itfc(std::shared_ptr<const _O> observation) ;

            bool has_child(std::shared_ptr<const _O> observation);
            shared_ptr<_DNode> get_child(shared_ptr<const _O> observation);

        /**
         * ThtsDNode interface function definitions. Boilerplate implementations provided from 
         * thts_chance_node_template.h
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
                std::shared_ptr<const Observation> observation);

            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation);
            // virtual std::shared_ptr<ThtsDNode> create_child_node_itfc(
            //    std::shared_ptr<const Observation> observation) final;
                
        /**
         * Implemented in thts_chance_node.cpp
         */
        // public:
        //     bool is_two_player_game();
        //     bool is_opponent();
        //     int get_num_children();

        //     bool has_child_itfc(std::shared_ptr<const Observation> observation);
        //     std::shared_ptr<ThtsDNode> get_child_node_itfc(std::shared_ptr<const Observation> observation);

        //     std::string get_pretty_print_string(int depth);

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
    _CNode::_CNode(
        shared_ptr<_Manager> thts_manager,
        shared_ptr<_Env> thts_env,
        shared_ptr<const _S> state,
        shared_ptr<const _A> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<ThtsDNode> parent) :
            ThtsCNode(
                static_pointer_cast<ThtsManager>(thts_manager),
                static_pointer_cast<ThtsEnv>(thts_env),
                static_pointer_cast<const State>(state),
                static_pointer_cast<const Action>(action),
                decision_depth,
                decision_timestep,
                static_pointer_cast<ThtsDNode>(parent)) {}

    void _CNode::visit(_Context& ctx) {
        num_visits += 1;
    }

    shared_ptr<const _O> _CNode::sample_observation(_Context& ctx) {
        return nullptr;
    }

    void _CNode::backup(
        const std::vector<double>& trial_rewards_before_node, 
        const std::vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        _Context& ctx)
    {   
    }

    shared_ptr<const _S> compute_next_state_from_observation(shared_ptr<const _O> observation) {
        return static_pointer_cast<const _S>(observation);
    }

    shared_ptr<_DNode> _CNode::create_child_node_helper(shared_ptr<const _O> observation) {
        shared_ptr<const _S> next_state = compute_next_state_from_observation(observation);
        return make_shared<_DNode>(
            thts_manager, 
            thts_env, 
            next_state,
            decision_depth+1, 
            decision_timestep+1, 
            this);
    }

    string get_pretty_print_val() {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<_DNode> _CNode::create_child_node_itfc(shared_ptr<const _O> observation) {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(action);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obsv_itfc);
        return static_pointer_cast<_DNode>(new_child);

    }

    bool _CNode::has_child(std::shared_ptr<const _O> observation) {
        return ThtsCNode::has_child_itfc(static_pointer_cast<const Observation>(observation));
    }
    
    shared_ptr<_DNode> _CNode::get_child(shared_ptr<const _O> observation) {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<_DNode>(new_child);
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {
    void _CNode::visit_itfc(ThtsEnvContext& ctx) {
        _Context& ctx_itfc = (_Context&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Observation> _CNode::sample_observation_itfc(ThtsEnvContext& ctx) {
        _Context& ctx_itfc = (_Context&) ctx;
        shared_ptr<const _O> obsv = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obsv);
    }

    void _CNode::backup_itfc(
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

    shared_ptr<const State> _CNode::compute_next_state_from_observation_itfc(
        shared_ptr<const Observation> observation)
    {
        shared_ptr<const _O> obsv_itfc = static_pointer_cast<const _O>(observation);
        shared_ptr<const _S> state = compute_next_state_from_observation(obsv_itfc);
        return static_pointer_cast<const State>(state);
    }

    shared_ptr<ThtsDNode> _CNode::create_child_node_helper_itfc(shared_ptr<const Observation> observation) {
        shared_ptr<const _O> obsv_itfc = static_pointer_cast<const _O>(observation);
        shared_ptr<_DNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);

    }
}