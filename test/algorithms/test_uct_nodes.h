#pragma once
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "algorithms/uct_chance_node.h"
#include "algorithms/uct_decision_node.h"
#include "algorithms/uct_manager.h"
#include "thts_env.h"
#include "thts_types.h"

#include <memory>


namespace thts_test {
    using namespace std;
    using namespace thts;

    // using ::testing::Action;
    using ::testing::ActionInterface;
    using ::testing::MakeAction;

    /**
     * Mocker for ThtsEnv
     */
    class MockThtsEnv_ForUct : public ThtsEnv {
        public:
            MockThtsEnv_ForUct() : ThtsEnv(true) {};

            MOCK_METHOD(shared_ptr<const State>, get_initial_state_itfc, (), (const, override));
            MOCK_METHOD(bool, is_sink_state_itfc, (shared_ptr<const State>), (const, override));
            MOCK_METHOD(shared_ptr<ActionVector>, get_valid_actions_itfc, (shared_ptr<const State>), (const, override));
            MOCK_METHOD(
                shared_ptr<StateDistr>, 
                get_transition_distribution_itfc, 
                (shared_ptr<const State>,shared_ptr<const Action>), 
                (const,override));
            MOCK_METHOD(
                shared_ptr<const State>, 
                sample_transition_distribution_itfc, 
                (shared_ptr<const State>,shared_ptr<const Action>,shared_ptr<ThtsManager> ), 
                (const,override));
            MOCK_METHOD(
                shared_ptr<ObservationDistr>, 
                get_observation_distribution_itfc, 
                (shared_ptr<const Action>,shared_ptr<const State>), 
                (const,override));
            MOCK_METHOD(
                shared_ptr<const Observation>, 
                sample_observation_distribution_itfc, 
                (shared_ptr<const Action>,shared_ptr<const State>,shared_ptr<ThtsManager> ), 
                (const,override));
            MOCK_METHOD(
                double, 
                get_reward_itfc, 
                (shared_ptr<const State>,shared_ptr<const Action>,shared_ptr<const Observation>),
                (const, override));
    };

    /**
     * Mock UctThtsManager. Used to spoof random number generation.
     */
    class MockUctManager : public UctManager {
        public:
            MockUctManager(shared_ptr<ThtsEnv> thts_env=nullptr, double bias=UctManager::USE_AUTO_BIAS) : 
                UctManager(thts_env) 
            {
                this->bias = bias;
            };

            MOCK_METHOD(int, get_rand_int, (int min_included, int max_excluded), (override));
            MOCK_METHOD(double, get_rand_uniform, (), (override));
    };

    /**
     * Adding getters and setters for protected variables in UctDNode
     * Sets state=nullptr, decisiondepth=decision_timestep=0 in ThtsDNode because they aren't relevant for uct tests
     */
    class SettableUctDNode : public UctDNode {
        public:
            SettableUctDNode(shared_ptr<UctManager> thts_manager) : 
                UctDNode(thts_manager, nullptr, 0, 0) {};
            
            shared_ptr<ActionVector> get_actions() { return actions; }
            void set_actions(shared_ptr<ActionVector> actions) { this->actions = actions; }
            double get_avg_return() { return avg_return; }
            void set_avg_return(double ret) { avg_return = ret; }
            shared_ptr<ActionPrior> get_policy_prior() { return policy_prior; }
            void set_policy_prior(shared_ptr<ActionPrior> prior) { policy_prior = prior; }
            CNodeChildMap get_children() { return children; }
            void set_children(CNodeChildMap child_map) { children = child_map; }

            // expose methods
            void fill_ucb_values(unordered_map<shared_ptr<const Action>,double>& ucb_values, ThtsEnvContext& ctx) const
            {
                UctDNode::fill_ucb_values(ucb_values, ctx);
            }
    };

    /**
     * Adding getters and setters for protected variables in UctCNode
     * Sets state=action=nullptr, decisiondepth=decision_timestep=0 in ThtsDNode because they aren't relevant for uct 
     * tests
     * 
     * Adds mock_avg_return and mock_num_visits in constructor for convenience
     */
    class SettableUctCNode : public UctCNode {
        public:
            SettableUctCNode(shared_ptr<UctManager> thts_manager, double mock_avg_return=0.0, int mock_num_visits=0) : 
                UctCNode(thts_manager, nullptr, nullptr, 0, 0) 
            {
                num_visits = mock_num_visits;
                num_backups = -mock_num_visits;
                avg_return = mock_avg_return;
            };
            
            shared_ptr<StateDistr> get_next_state_distr() { return next_state_distr; }
            void set_next_state_distr(shared_ptr<StateDistr> distr) { next_state_distr = distr; }
            double get_avg_return() { return avg_return; }
            void set_avg_return(double ret) { avg_return = ret; }
    };

    /**
     * A mock version of UctDNode for testing 'fill_ucb_values'
     */
    class MockUctDNode_ComputeUcbMock : public SettableUctDNode {
        public:
            MockUctDNode_ComputeUcbMock(shared_ptr<UctManager> thts_manager) : SettableUctDNode(thts_manager) {};

            MOCK_METHOD(double, compute_ucb_term, (int,int),  (const, override));
            MOCK_METHOD(bool, is_opponent, (), (const, override));
    };

    /**
     * Implement an gmock action interface for being able to spoof 'fill_ucb_values'.
     * 
     * The action class stores the value map, and implements the 'Perform' function to fill the ucb values that will be 
     * passed into 'fill_ucb_values'. 
     * 
     * Finally, we create a function that creates the action so that it can be used in EXPECT_CALL actions.
     * Note that ::testing::Action and thts::Action overload each other, so need to be careful.
     */
    typedef unordered_map<shared_ptr<const Action>,double> UcbValueMap;
    typedef void FillUcbValuesType(UcbValueMap&, ThtsEnvContext&);

    class MockFillUcbValuesAction : public ActionInterface<FillUcbValuesType> {
        public:
            UcbValueMap stored_values;
            MockFillUcbValuesAction(UcbValueMap& ucb_values) : 
                ActionInterface<FillUcbValuesType>(), stored_values(ucb_values) {}

            void Perform(const std::tuple<UcbValueMap&, ThtsEnvContext&>& args) override {
                UcbValueMap& ucb_values = std::get<0>(args);
                for (std::pair<shared_ptr<const thts::Action>,double> pr : stored_values) {
                    ucb_values[pr.first] = pr.second;
                }
            }
    };

    ::testing::Action<FillUcbValuesType> FillUcbValues(UcbValueMap& ucb_values) {
        return MakeAction(new MockFillUcbValuesAction(ucb_values));
    }

    /**
     * A mock version of UctDNode for testing 'select_action' and 'recommend_action'. 
     * 
     * Also exposes protected methods that we want to test directly.
     */
    class MockUctDNode_SelectActionMock : public SettableUctDNode {
        public:
            MockUctDNode_SelectActionMock(shared_ptr<UctManager> thts_manager) : SettableUctDNode(thts_manager) {};

            MOCK_METHOD(bool, has_prior, (), (const, override));

            MOCK_METHOD(
                void, 
                fill_ucb_values, 
                ((unordered_map<shared_ptr<const Action>,double>&), ThtsEnvContext&), 
                (const, override));

            // expose methods
            std::shared_ptr<const Action> select_action_ucb(ThtsEnvContext& ctx) {
                return UctDNode::select_action_ucb(ctx);
            }

            std::shared_ptr<const Action> select_action_random() {
                return UctDNode::select_action_random();
            }
    };

    /**
     * Second mock version of Uct D Node, to test epsilon exploration
     */
    class MockUctDNode_SelectActionMockTwo : public SettableUctDNode {
        public:
            MockUctDNode_SelectActionMockTwo(shared_ptr<UctManager> thts_manager) : SettableUctDNode(thts_manager) {};

            MOCK_METHOD(
                std::shared_ptr<const Action>, 
                select_action_ucb,
                (ThtsEnvContext& ctx),
                (override));
            
            MOCK_METHOD(
                std::shared_ptr<const Action>, 
                select_action_random,
                (),
                (override));
    };
}