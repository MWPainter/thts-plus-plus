#include "test_thts_nodes.h"

#include "test_thts_env.h"

#include "thts_chance_node.h"
#include "thts_decision_node.h"
#include "thts_manager.h"

#include <iostream>
#include <memory>
#include <string>

using namespace std;
using namespace thts;

namespace thts_test{
    // Forward declare
    class TestThtsCNode;

    /**
     * 
     */
    class TestThtsDNode : public ThtsDNode {
        public:
            TestThtsDNode(
                shared_ptr<ThtsManager> thts_manager,
                shared_ptr<ThtsEnv> thts_env,
                shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep) :
                    ThtsDNode(thts_manager,thts_env,state,decision_depth,decision_timestep) {}

            shared_ptr<const Action> select_action_itfc(ThtsEnvContext& ctx) { return nullptr; }

            shared_ptr<const Action> recommend_action_itfc(ThtsEnvContext& ctx) { return nullptr; }

            void backup_itfc(
                    const vector<double>& trial_rewards_before_node, 
                    const vector<double>& trial_rewards_after_node, 
                    const double trial_cumulative_return_after_node, 
                    const double trial_cumulative_return,
                    ThtsEnvContext& ctx) {}

            bool is_leaf() { return false; }
        
        protected:
            shared_ptr<ThtsCNode> create_child_node_helper_itfc(shared_ptr<const Action> action) { 
                shared_ptr<TestThtsCNode> new_child_ptr = make_shared<TestThtsCNode>(
                    thts_manager,thts_env,state,action,decision_depth,decision_timestep);
                return static_pointer_cast<ThtsCNode>(new_child_ptr); 
            }

            string get_pretty_print_val() { return "0.0"; }
    };

    /**
     * 
     */
    class TestThtsCNode : public ThtsCNode {
        public: 
            TestThtsCNode(
                shared_ptr<ThtsManager> thts_manager,
                shared_ptr<ThtsEnv> thts_env,
                shared_ptr<const State> state,
                shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep) :
                    ThtsCNode(thts_manager,thts_env,state,action,decision_depth,decision_timestep) {}

            shared_ptr<const Observation> sample_observation_itfc(ThtsEnvContext& ctx) { return nullptr; }

            void backup_itfc(
                    const vector<double>& trial_rewards_before_node, 
                    const vector<double>& trial_rewards_after_node, 
                    const double trial_cumulative_return_after_node, 
                    const double trial_cumulative_return,
                    ThtsEnvContext& ctx) {}

            shared_ptr<const State> compute_next_state_from_observation_itfc(shared_ptr<const Observation> observation) {
                return static_pointer_cast<const State>(observation);
            }

        protected:
            shared_ptr<ThtsDNode> create_child_node_helper_itfc(shared_ptr<const Observation> observation) { 
                shared_ptr<const State> next_state = compute_next_state_from_observation_itfc(observation);
                shared_ptr<TestThtsDNode> new_child_ptr = make_shared<TestThtsDNode>(
                    thts_manager,thts_env,next_state,decision_depth+1,decision_timestep+1);
                return new_child_ptr;
            }

            string get_pretty_print_val() { return "0.0"; }
    };

    /**
     * 
     */
    void run_thts_node_tests() {
        cout << "----------" << endl;
        cout << "Testing ThtsDNode and ThtsCNode using test subclasses." << endl;
        cout << "----------" << endl;

        ThtsManager thts_manager;
        shared_ptr<ThtsEnv> thts_env = static_pointer_cast<ThtsEnv>(make_shared<TestThtsEnv>(2));
        shared_ptr<TestThtsDNode> root_node = make_shared<TestThtsDNode>(
            make_shared<ThtsManager>(thts_manager),
            thts_env,
            thts_env->get_initial_state_itfc(),
            0,
            0);
        shared_ptr<const Action> act = static_pointer_cast<const Action>(
            make_shared<const StringAction>("right"));
        shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(
            make_shared<const IntPairState>(1,0));
        
        shared_ptr<TestThtsCNode> r_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
        shared_ptr<TestThtsDNode> r_node = static_pointer_cast<TestThtsDNode>(r_cnode->create_child_node_itfc(obsv));

        act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
        obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

        shared_ptr<TestThtsCNode> rd_cnode = static_pointer_cast<TestThtsCNode>(r_node->create_child_node_itfc(act));
        shared_ptr<TestThtsDNode> rd_node = static_pointer_cast<TestThtsDNode>(rd_cnode->create_child_node_itfc(obsv));


        act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
        obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(0,1));

        shared_ptr<TestThtsCNode> d_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
        shared_ptr<TestThtsDNode> d_node = static_pointer_cast<TestThtsDNode>(d_cnode->create_child_node_itfc(obsv));

        act = static_pointer_cast<const Action>(make_shared<const StringAction>("right"));
        obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

        shared_ptr<TestThtsCNode> dr_cnode = static_pointer_cast<TestThtsCNode>(d_node->create_child_node_itfc(act));
        shared_ptr<TestThtsDNode> dr_node = static_pointer_cast<TestThtsDNode>(dr_cnode->create_child_node_itfc(obsv));

        ThtsEnvContext ctx;
        r_cnode->visit_itfc(ctx);
        r_node->visit_itfc(ctx);
        rd_cnode->visit_itfc(ctx);
        rd_node->visit_itfc(ctx);
        d_cnode->visit_itfc(ctx);
        d_node->visit_itfc(ctx);
        dr_cnode->visit_itfc(ctx);
        dr_node->visit_itfc(ctx);


        cout << "Testing pretty print, with two paths to (1,1) without transposition table:" << endl;
        cout << root_node->get_pretty_print_string(10) << endl << endl;

        shared_ptr<ThtsManager> new_thts_manager = make_shared<ThtsManager>();
        new_thts_manager->use_transposition_table = true;
        root_node = make_shared<TestThtsDNode>(
            new_thts_manager,
            thts_env,
            thts_env->get_initial_state_itfc(),
            0,
            0);

        act = static_pointer_cast<const Action>(make_shared<const StringAction>("right"));
        obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,0));
        
        r_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
        r_node = static_pointer_cast<TestThtsDNode>(r_cnode->create_child_node_itfc(obsv));

        act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
        obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

        rd_cnode = static_pointer_cast<TestThtsCNode>(r_node->create_child_node_itfc(act));
        rd_node = static_pointer_cast<TestThtsDNode>(rd_cnode->create_child_node_itfc(obsv));

        act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
        obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(0,1));

        d_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
        d_node = static_pointer_cast<TestThtsDNode>(d_cnode->create_child_node_itfc(obsv));

        act = static_pointer_cast<const Action>(make_shared<const StringAction>("right"));
        obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

        dr_cnode = static_pointer_cast<TestThtsCNode>(d_node->create_child_node_itfc(act));
        dr_node = static_pointer_cast<TestThtsDNode>(dr_cnode->create_child_node_itfc(obsv));

        // ThtsEnvContext ctx;
        r_cnode->visit_itfc(ctx);
        r_node->visit_itfc(ctx);
        rd_cnode->visit_itfc(ctx);
        rd_node->visit_itfc(ctx);
        d_cnode->visit_itfc(ctx);
        d_node->visit_itfc(ctx);
        dr_cnode->visit_itfc(ctx);
        dr_node->visit_itfc(ctx);

        cout << "Testing pretty print, with two paths to (1,1) WITH transposition table:" << endl;
        cout << root_node->get_pretty_print_string(10) << endl << endl;
    }
}