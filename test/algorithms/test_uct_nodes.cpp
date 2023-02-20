#include "test/algorithms/test_uct_nodes.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "algorithms/uct_chance_node.h"
#include "algorithms/uct_decision_node.h"
#include "algorithms/uct_manager.h"

// includes
#include "test/test_thts_env.h"
#include "thts.h"
#include "thts_env_context.h"

#include <iostream>
#include <unordered_map>


using namespace std;
using namespace thts;
using namespace thts_test;

// actions (for 'EXPECT_CALL')
using ::testing::Return;


/**
 * Test when we call 'fill_ucb_values' without a prior. Note that this should never be called when 
 * children.size() != actions.size(), so we don't test for that case.
 */
TEST(Uct_Ucb, compute_ucb_values_no_prior) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    uct_manager->bias = 2.0;
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Children
    CNodeChildMap children;
    children[a[0]] = make_shared<SettableUctCNode>(uct_manager, 1.0);
    children[a[1]] = make_shared<SettableUctCNode>(uct_manager, 0.5);
    children[a[2]] = make_shared<SettableUctCNode>(uct_manager, 1.0);

    // Uct D Node
    shared_ptr<MockUctDNode_ComputeUcbMock> uct_d_node = make_shared<MockUctDNode_ComputeUcbMock>(uct_manager);
    EXPECT_CALL(*uct_d_node, compute_ucb_term)
        .Times(3)
        .WillOnce(Return(1.0))
        .WillRepeatedly(Return(3.0));
    EXPECT_CALL(*uct_d_node, is_opponent)
        .Times(1)
        .WillOnce(Return(false));
    // uct_d_node->set_actions(actions);
    uct_d_node->set_children(children);

    // Fill ucb values
    unordered_map<shared_ptr<const Action>,double> ucb_values;
    uct_d_node->fill_ucb_values(ucb_values, *dummy_context);

    // Checks
    EXPECT_EQ(ucb_values.size(), 3u);
    EXPECT_EQ(ucb_values[a[0]], 1.0+2.0*1.0);
    EXPECT_EQ(ucb_values[a[1]], 0.5+2.0*3.0);
    EXPECT_EQ(ucb_values[a[2]], 1.0+2.0*3.0);
}

/**
 * Test when we call 'fill_ucb_values' with a prior. Note that this can be called when 
 * children.size() != actions.size(), so we make sure to cover this case.
 */
shared_ptr<ActionPrior> mock_prior_fn(shared_ptr<const State> state, shared_ptr<ThtsEnv> env=nullptr) {
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Policy prior
    shared_ptr<ActionPrior> policy_prior = make_shared<ActionPrior>();
    ActionPrior& prior = *policy_prior;
    prior[a[0]] = 0.8;
    prior[a[1]] = 0.1;
    prior[a[2]] = 0.1;

    return policy_prior;
}

TEST(Uct_Ucb, compute_ucb_values_with_prior) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    uct_manager->bias = 2.0;
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Policy prior
    shared_ptr<ActionPrior> policy_prior = make_shared<ActionPrior>();
    ActionPrior& prior = *policy_prior;
    prior[a[0]] = 0.8;
    prior[a[1]] = 0.1;
    prior[a[2]] = 0.1;

    uct_manager->prior_fn = mock_prior_fn;

    // Children
    CNodeChildMap children;
    // children[a[0]] = make_shared<SettableUctCNode>(uct_manager, 1.0);
    children[a[1]] = make_shared<SettableUctCNode>(uct_manager, 0.5);
    children[a[2]] = make_shared<SettableUctCNode>(uct_manager, 1.0);

    // Uct D Node
    shared_ptr<MockUctDNode_ComputeUcbMock> uct_d_node = make_shared<MockUctDNode_ComputeUcbMock>(uct_manager);
    EXPECT_CALL(*uct_d_node, compute_ucb_term)
        .Times(3)
        .WillOnce(Return(1.0))
        .WillRepeatedly(Return(3.0));
    EXPECT_CALL(*uct_d_node, is_opponent)
        .Times(1)
        .WillOnce(Return(false));
    // uct_d_node->set_actions(actions);
    // uct_d_node->set_prior(policy_prior)
    uct_d_node->set_children(children);

    // Fill ucb values
    unordered_map<shared_ptr<const Action>,double> ucb_values;
    uct_d_node->fill_ucb_values(ucb_values, *dummy_context);

    // Checks
    EXPECT_EQ(ucb_values.size(), 3u);
    EXPECT_EQ(ucb_values[a[0]], 0.0+0.8*2.0*1.0);
    EXPECT_EQ(ucb_values[a[1]], 0.5+0.1*2.0*3.0);
    EXPECT_EQ(ucb_values[a[2]], 1.0+0.1*2.0*3.0);
}

/**
 * Test when we call 'fill_ucb_values' without a prior as an opponent. Note that this should never be called when 
 * children.size() != actions.size(), so we don't test for that case.
 * 
 * Note that we still want to add a positive confidence interval, but the q-values are negated.
 */
TEST(Uct_Ucb, compute_ucb_values_opponent) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    uct_manager->bias = 2.0;
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Children
    CNodeChildMap children;
    children[a[0]] = make_shared<SettableUctCNode>(uct_manager, 1.0);
    children[a[1]] = make_shared<SettableUctCNode>(uct_manager, 0.5);
    children[a[2]] = make_shared<SettableUctCNode>(uct_manager, 1.0);

    // Uct D Node
    shared_ptr<MockUctDNode_ComputeUcbMock> uct_d_node = make_shared<MockUctDNode_ComputeUcbMock>(uct_manager);
    EXPECT_CALL(*uct_d_node, compute_ucb_term)
        .Times(3)
        .WillOnce(Return(1.0))
        .WillRepeatedly(Return(3.0));
    EXPECT_CALL(*uct_d_node, is_opponent)
        .Times(1)
        .WillOnce(Return(true));
    // uct_d_node->set_actions(actions);
    uct_d_node->set_children(children);

    // Fill ucb values
    unordered_map<shared_ptr<const Action>,double> ucb_values;
    uct_d_node->fill_ucb_values(ucb_values, *dummy_context);

    // Checks
    EXPECT_EQ(ucb_values.size(), 3u);
    EXPECT_EQ(ucb_values[a[0]], -1.0+2.0*1.0);
    EXPECT_EQ(ucb_values[a[1]], -0.5+2.0*3.0);
    EXPECT_EQ(ucb_values[a[2]], -1.0+2.0*3.0);
}


/**
 * Test that ucb select action creates new children when they dont exist yet
 */
TEST(Uct_SelectAction, uct_actions_yet_to_sample_no_prior) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);

    // Mock random number generation
    EXPECT_CALL(*uct_manager, get_rand_int)
        .Times(2)
        .WillOnce(Return(1))
        .WillOnce(Return(0));
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Children
    CNodeChildMap children;
    children[a[0]] = make_shared<SettableUctCNode>(uct_manager, 1.0);
    shared_ptr<SettableUctCNode> child_a1 = make_shared<SettableUctCNode>(uct_manager, 0.5);
    shared_ptr<SettableUctCNode> child_a2 = make_shared<SettableUctCNode>(uct_manager, 1.0);

    // Uct D Node
    shared_ptr<MockUctDNode_SelectActionMock> uct_d_node = make_shared<MockUctDNode_SelectActionMock>(uct_manager);
    EXPECT_CALL(*uct_d_node, fill_ucb_values)
        .Times(0);
    EXPECT_CALL(*uct_d_node, has_prior)
        .Times(2)
        .WillRepeatedly(Return(false));
    // uct_d_node->set_actions(actions);
    uct_d_node->set_children(children);

    // Check only one child
    EXPECT_EQ(uct_d_node->get_num_children(), 1);

    // Run select action ucb
    shared_ptr<const Action> first_action = uct_d_node->select_action_ucb(*dummy_context);
    EXPECT_EQ(first_action, a[2]);
    EXPECT_EQ(uct_d_node->get_num_children(), 2);
    EXPECT_FALSE(uct_d_node->has_child_node(a[1]));
    EXPECT_TRUE(uct_d_node->has_child_node(a[2]));

    // Anotha one
    shared_ptr<const Action> second_action = uct_d_node->select_action_ucb(*dummy_context);
    EXPECT_EQ(second_action, a[1]);
    EXPECT_EQ(uct_d_node->get_num_children(), 3);
    EXPECT_TRUE(uct_d_node->has_child_node(a[1]));
    EXPECT_TRUE(uct_d_node->has_child_node(a[2]));
}

/**
 * Checks that when children is filled, that select action ucb picks the highest ucb value
 */
TEST(Uct_SelectAction, uct_all_actions_previously_chosen) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);

    // Mock random number generation
    EXPECT_CALL(*uct_manager, get_rand_int)
        .Times(0);
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Children
    CNodeChildMap children;
    children[a[0]] = make_shared<SettableUctCNode>(uct_manager, 1.0);
    children[a[1]] = make_shared<SettableUctCNode>(uct_manager, 0.5);
    children[a[2]] = make_shared<SettableUctCNode>(uct_manager, 1.0);

    // The two mock ucb values that we will use
    UcbValueMap mock_ucb_values_one = {{a[0], 0.0}, {a[1], 1.0}, {a[2], 2.0}};
    UcbValueMap mock_ucb_values_two = {{a[0], 9.0}, {a[1], 2.0}, {a[2], 0.0}};

    // Uct D Node
    shared_ptr<MockUctDNode_SelectActionMock> uct_d_node = make_shared<MockUctDNode_SelectActionMock>(uct_manager);
    EXPECT_CALL(*uct_d_node, fill_ucb_values)
        .Times(2)
        .WillOnce(FillUcbValues(mock_ucb_values_one))
        .WillOnce(FillUcbValues(mock_ucb_values_two));
    EXPECT_CALL(*uct_d_node, has_prior)
        .Times(2)
        .WillRepeatedly(Return(false));
    // uct_d_node->set_actions(actions);
    uct_d_node->set_children(children);

    // Expect all 3 children
    EXPECT_EQ(uct_d_node->get_num_children(), 3);

    // Run two select action calls, and 
    // check that they correspond to a[2] having a value of 2.0 the first time
    // and correspond to a[0] having a value of 9.0 the second time
    shared_ptr<const Action> first_action = uct_d_node->select_action_ucb(*dummy_context);
    EXPECT_EQ(first_action, a[2]);
    shared_ptr<const Action> second_action = uct_d_node->select_action_ucb(*dummy_context);
    EXPECT_EQ(second_action, a[0]);
}

/**
 * Checks that when a prior is present, that we always use fill_ucb_values, and that new nodes are created when they 
 * are selected
 */
TEST(Uct_SelectAction, uct_with_prior) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);

    // Mock random number generation
    EXPECT_CALL(*uct_manager, get_rand_int)
        .Times(0);
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Children
    CNodeChildMap children;
    shared_ptr<SettableUctCNode> c0 = make_shared<SettableUctCNode>(uct_manager, 1.0);
    children[a[1]] = make_shared<SettableUctCNode>(uct_manager, 0.5);
    children[a[2]] = make_shared<SettableUctCNode>(uct_manager, 1.0);

    // The two mock ucb values that we will use
    UcbValueMap mock_ucb_values_one = {{a[0], 0.0}, {a[1], 1.0}, {a[2], 2.0}};
    UcbValueMap mock_ucb_values_two = {{a[0], 9.0}, {a[1], 2.0}, {a[2], 0.0}};

    // Uct D Node
    shared_ptr<MockUctDNode_SelectActionMock> uct_d_node = make_shared<MockUctDNode_SelectActionMock>(uct_manager);
    EXPECT_CALL(*uct_d_node, fill_ucb_values)
        .Times(2)
        .WillOnce(FillUcbValues(mock_ucb_values_one))
        .WillOnce(FillUcbValues(mock_ucb_values_two));
    EXPECT_CALL(*uct_d_node, has_prior)
        .Times(2)
        .WillRepeatedly(Return(true));
    // uct_d_node->set_actions(actions);
    uct_d_node->set_children(children);

    // Expect two children
    EXPECT_EQ(uct_d_node->get_num_children(), 2);

    // First action call, action should be a[2] (value of 2.0), which corresponds to a node that already exists
    shared_ptr<const Action> first_action = uct_d_node->select_action_ucb(*dummy_context);
    EXPECT_EQ(first_action, a[2]);
    EXPECT_EQ(uct_d_node->get_num_children(), 2);
    EXPECT_FALSE(uct_d_node->has_child_node(a[0]));

    // Secon action call, action should be a[0] (value of 9.0), which should create the node
    shared_ptr<const Action> second_action = uct_d_node->select_action_ucb(*dummy_context);
    EXPECT_EQ(second_action, a[0]);
    EXPECT_EQ(uct_d_node->get_num_children(), 3);
    EXPECT_TRUE(uct_d_node->has_child_node(a[0]));
}

/**
 * Tests 'select_action_random', and that it creates children when new actions are sampled.
 */
TEST(Uct_SelectAction, random_action) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);

    // Mock random number generation
    EXPECT_CALL(*uct_manager, get_rand_int)
        .Times(2)
        .WillOnce(Return(2))
        .WillOnce(Return(0));
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Children
    CNodeChildMap children;
    shared_ptr<SettableUctCNode> c0 = make_shared<SettableUctCNode>(uct_manager, 1.0);
    children[a[1]] = make_shared<SettableUctCNode>(uct_manager, 0.5);
    children[a[2]] = make_shared<SettableUctCNode>(uct_manager, 1.0);

    // Uct D Node
    shared_ptr<MockUctDNode_SelectActionMock> uct_d_node = make_shared<MockUctDNode_SelectActionMock>(uct_manager);
    EXPECT_CALL(*uct_d_node, fill_ucb_values)
        .Times(0);
    EXPECT_CALL(*uct_d_node, has_prior)
        .Times(0);
    // uct_d_node->set_actions(actions);
    uct_d_node->set_children(children);

    // Expect two children
    EXPECT_EQ(uct_d_node->get_num_children(), 2);

    // First action call, action should be a[2] (value of 2.0), which corresponds to a node that already exists
    shared_ptr<const Action> first_action = uct_d_node->select_action_random();
    EXPECT_EQ(first_action, a[2]);
    EXPECT_EQ(uct_d_node->get_num_children(), 2);
    EXPECT_FALSE(uct_d_node->has_child_node(a[0]));

    // Secon action call, action should be a[0] (value of 9.0), which should create the node
    shared_ptr<const Action> second_action = uct_d_node->select_action_random();
    EXPECT_EQ(second_action, a[0]);
    EXPECT_EQ(uct_d_node->get_num_children(), 3);
    EXPECT_TRUE(uct_d_node->has_child_node(a[0]));
}

TEST(Uct_SelectAction, epsilon_exploration) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    uct_manager->epsilon_exploration = 0.0;
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);

    // Mock random number generation
    EXPECT_CALL(*uct_manager, get_rand_uniform)
        .Times(2)
        .WillOnce(Return(0.25))
        .WillOnce(Return(0.75));
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Uct D Node
    shared_ptr<MockUctDNode_SelectActionMockTwo> uct_d_node = make_shared<MockUctDNode_SelectActionMockTwo>(
        uct_manager);
    EXPECT_CALL(*uct_d_node, select_action_ucb)
        .Times(2)
        .WillRepeatedly(Return(a[0]));
    EXPECT_CALL(*uct_d_node, select_action_random)
        .Times(1)
        .WillRepeatedly(Return(a[1]));

    // Call select action, should just call select_action_ucb directly as epsilon exploration == 0
    // And shouldn't sample any random numbers
    uct_d_node->select_action(*dummy_context);

    // Set epsilon exploration to 0.5, and mocker should force each option (ucb/random) to be called once
    uct_manager->epsilon_exploration = 0.5;
    uct_d_node->select_action(*dummy_context);
    uct_d_node->select_action(*dummy_context);
}



/**
 * Test that recommend action returns the empirical best by default
 */
TEST(Uct_RecommendAction, empirical_best) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);

    // Mock random number generation
    EXPECT_CALL(*uct_manager, get_rand_int)
        .Times(0);
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Children (a2 most visited, a1 best value)
    CNodeChildMap children;
    children[a[0]] = make_shared<SettableUctCNode>(uct_manager, 100.0, 555);
    children[a[1]] = make_shared<SettableUctCNode>(uct_manager, 456.5, 111);
    children[a[2]] = make_shared<SettableUctCNode>(uct_manager, -99.0, 999);

    // Uct D Node
    shared_ptr<MockUctDNode_SelectActionMock> uct_d_node = make_shared<MockUctDNode_SelectActionMock>(uct_manager);
    EXPECT_CALL(*uct_d_node, fill_ucb_values)
        .Times(0);
    EXPECT_CALL(*uct_d_node, has_prior)
        .Times(0);
    // uct_d_node->set_actions(actions);
    uct_d_node->set_children(children);

    // Expect to recommend a[1]
    EXPECT_EQ(uct_d_node->recommend_action(*dummy_context), a[1]);
}

/**
 * Test that recommend action returns the most visited when UctManager option is set
 * (Only difference to the above test is the line 'uct_manager->recommend_most_visited = true;')
 */
TEST(Uct_RecommendAction, most_visited) {
    shared_ptr<MockThtsEnv_ForUct> mock_env = make_shared<MockThtsEnv_ForUct>();
    shared_ptr<MockUctManager> uct_manager = make_shared<MockUctManager>(mock_env);
    uct_manager->recommend_most_visited = true;
    shared_ptr<ThtsEnvContext> dummy_context = mock_env->sample_context_itfc(nullptr);

    // Mock random number generation
    EXPECT_CALL(*uct_manager, get_rand_int)
        .Times(0);
    
    // Actions
    shared_ptr<ActionVector> actions = make_shared<ActionVector>(3);
    ActionVector& a = *actions;
    a[0] = make_shared<IntAction>(0);
    a[1] = make_shared<IntAction>(1);
    a[2] = make_shared<IntAction>(2);

    // Make mock env give actions to Uct D Node when make it
    EXPECT_CALL(*mock_env, get_valid_actions_itfc)
        .Times(1)
        .WillOnce(Return(actions));

    // Children (a2 most visited, a1 best value)
    CNodeChildMap children;
    children[a[0]] = make_shared<SettableUctCNode>(uct_manager, 100.0, 555);
    children[a[1]] = make_shared<SettableUctCNode>(uct_manager, 456.5, 111);
    children[a[2]] = make_shared<SettableUctCNode>(uct_manager, -99.0, 999);

    // Uct D Node
    shared_ptr<MockUctDNode_SelectActionMock> uct_d_node = make_shared<MockUctDNode_SelectActionMock>(uct_manager);
    EXPECT_CALL(*uct_d_node, fill_ucb_values)
        .Times(0);
    EXPECT_CALL(*uct_d_node, has_prior)
        .Times(0);
    // uct_d_node->set_actions(actions);
    uct_d_node->set_children(children);

    // Expect to recommend a[1]
    EXPECT_EQ(uct_d_node->recommend_action(*dummy_context), a[2]);
}



// Dummy test :)
TEST(Uct_ChanceNode, not_much_to_test) {
    // Dummy test to say "I've looked at it". UctCNode mostly just calls other functions, that should be tested 
    // elsewhere
}



/**
 * Actually run the full whack on a simple env and test it all works
 * 
 * Prints some fun things out for if we want to read
 */
void run_uct_integration_test(int env_size, int num_threads, int num_trials, double stay_prob=0.0, int print_tree_depth=0) {
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();

    shared_ptr<ThtsEnv> grid_env = make_shared<TestThtsEnv>(env_size, stay_prob);
    shared_ptr<UctManager> manager = make_shared<UctManager>(grid_env, env_size*4);
    manager->mcts_mode = false;
    shared_ptr<UctDNode> root_node = make_shared<UctDNode>(manager, grid_env->get_initial_state_itfc(), 0, 0);
    ThtsPool uct_pool(manager, root_node, num_threads);
    uct_pool.run_trials(num_trials);

    if (stay_prob == 0.0) {
        // TODO add asserts
    }

    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    cout << "UCT with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0){
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    } else {
        cout << endl;
    }
}

TEST(Uct_IntegrationTest, easy_grid_world) {
    run_uct_integration_test(1,1,10000,0.0,2);
}

TEST(Uct_IntegrationTest, easy_grid_world_multithreaded) {
    run_uct_integration_test(2,4,10000,0.0,1);
}

TEST(Uct_IntegrationTest, easy_grid_world_stochastic) {
    run_uct_integration_test(1,1,10000,0.1,2);
}

TEST(Uct_IntegrationTest, easy_grid_world_stochastic_multithreaded) {
    run_uct_integration_test(2,4,10000,0.1,1);
}


// TODO: Add test that check for #trials == #nodes in mcts mode
TEST(Uct_IntegrationTest, mcts_mode_todo) {
    FAIL();
}



/**
 * Also run full whack on a simple game to check that the opponent logic all works
 */
void run_uct_game_integration_test(int env_size, int num_trials, int print_tree_depth=0, int decision_timestep=0) {
    shared_ptr<ThtsEnv> game_env = make_shared<TestThtsGameEnv>(env_size);
    shared_ptr<UctManager> manager = make_shared<UctManager>(game_env, env_size*4);
    manager->mcts_mode = false;
    manager->is_two_player_game = true;
    shared_ptr<UctDNode> root_node = make_shared<UctDNode>(
        manager, game_env->get_initial_state_itfc(), 0, decision_timestep);
    ThtsPool uct_pool(manager, root_node, 1);
    uct_pool.run_trials(num_trials);

    if (print_tree_depth > 0){
        cout << "UCT with starting decision_timestep of " << decision_timestep << " looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    }
}

TEST(Uct_IntegrationTest, two_player_game_env) {
    run_uct_game_integration_test(3, 10000, 4, 0);
}

TEST(Uct_IntegrationTest, two_player_game_env_starting_as_opponent) {
    run_uct_game_integration_test(3, 10000, 4, 1);
}