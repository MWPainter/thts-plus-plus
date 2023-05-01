#pragma once

#include "thts_chance_node.h"
#include "thts_decision_node.h"
#include "thts_env_context.h"
#include "thts_manager.h"



namespace thts::test{
    using namespace std;
    using namespace thts;

    // Forward declare
    class TestThtsCNode;

    /**
     * A barebones mock DNode to be able to test the implementation that exists in thts_decision_node.{h,cpp}
     * 
     * Adding some getters to be able to check the protected state of the node too.
     */
    class TestThtsDNode : public ThtsDNode {
        public:
            TestThtsDNode(
                shared_ptr<ThtsManager> thts_manager,
                shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep) :
                    ThtsDNode(thts_manager,state,decision_depth,decision_timestep) {}

            shared_ptr<const Action> select_action_itfc(ThtsEnvContext& ctx) { return nullptr; }

            shared_ptr<const Action> recommend_action_itfc(ThtsEnvContext& ctx) const { return nullptr; }

            void backup_itfc(
                    const vector<double>& trial_rewards_before_node, 
                    const vector<double>& trial_rewards_after_node, 
                    const double trial_cumulative_return_after_node, 
                    const double trial_cumulative_return,
                    ThtsEnvContext& ctx) {}

            bool is_leaf() { return false; }

            // getters
            int get_num_visits() const { return num_visits; }
            int get_decision_depth() const { return decision_depth; }
            int get_decision_timestep() const { return decision_timestep; }
        
        protected:
            shared_ptr<ThtsCNode> create_child_node_helper_itfc(shared_ptr<const Action> action) const { 
                shared_ptr<TestThtsCNode> new_child_ptr = make_shared<TestThtsCNode>(
                    thts_manager,state,action,decision_depth,decision_timestep);
                return static_pointer_cast<ThtsCNode>(new_child_ptr); 
            }

            string get_pretty_print_val() const { return "0.0"; }
    };

    /**
     * A barebones mock DNode to be able to test the implementation that exists in thts_decision_node.{h,cpp}
     * 
     * Adding some getters to be able to check the protected state of the node too.
     */
    class TestThtsCNode : public ThtsCNode {
        public: 
            TestThtsCNode(
                shared_ptr<ThtsManager> thts_manager,
                shared_ptr<const State> state,
                shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep) :
                    ThtsCNode(thts_manager,state,action,decision_depth,decision_timestep) {}

            shared_ptr<const Observation> sample_observation_itfc(ThtsEnvContext& ctx) { return nullptr; }

            void backup_itfc(
                    const vector<double>& trial_rewards_before_node, 
                    const vector<double>& trial_rewards_after_node, 
                    const double trial_cumulative_return_after_node, 
                    const double trial_cumulative_return,
                    ThtsEnvContext& ctx) {}

            shared_ptr<const State> compute_next_state_from_observation_itfc(
                shared_ptr<const Observation> observation) const 
            {
                return static_pointer_cast<const State>(observation);
            }

            // getters
            int get_num_visits() const { return num_visits; }
            int get_decision_depth() const { return decision_depth; }
            int get_decision_timestep() const { return decision_timestep; }

        protected:
            shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                shared_ptr<const Observation> observation, shared_ptr<const State> next_state=nullptr) const 
            { 
                shared_ptr<const State> mdp_next_state = compute_next_state_from_observation_itfc(observation);
                shared_ptr<TestThtsDNode> new_child_ptr = make_shared<TestThtsDNode>(
                    thts_manager,mdp_next_state,decision_depth+1,decision_timestep+1);
                return static_pointer_cast<ThtsDNode>(new_child_ptr);
            }

            string get_pretty_print_val() const { return "0.0"; }
    };
}