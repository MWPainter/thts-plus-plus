#include "test_thts_nodes.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "thts_chance_node.h"
#include "thts_decision_node.h"

// includes
#include "test_thts_env.h"
#include "test_thts_manager.h"

#include "thts_manager.h"

#include <memory>
#include <string>


using namespace std;
using namespace thts;
using namespace thts_test;

// matchers (for assertations)
using ::testing::StrEq;




/**
 * Test normal creation of nodes and that they get setup correctly.
 */
TEST(ThtsNode_CreateChild, test_normal_usage)
{   
    // Mock manager
    shared_ptr<MockThtsManager> mock_manager_ptr = make_shared<MockThtsManager>();
    MockThtsManager& mock_manager = *mock_manager_ptr;
    shared_ptr<ThtsManager> manager_ptr = static_pointer_cast<ThtsManager>(mock_manager_ptr);
    EXPECT_CALL(mock_manager, get_rand_int)
        .Times(0);
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(0);
    
    // Mock decision depth and decision timestep
    int mock_decision_depth = 0;
    int mock_decision_timestep = 21;

    //Make env and root node
    shared_ptr<ThtsEnv> thts_env = static_pointer_cast<ThtsEnv>(make_shared<TestThtsEnv>(2));
    shared_ptr<TestThtsDNode> root_node = make_shared<TestThtsDNode>(
        manager_ptr,
        thts_env,
        thts_env->get_initial_state_itfc(),
        mock_decision_depth,
        mock_decision_timestep);

    // Actions, observation pair for making a (r) child
    shared_ptr<const Action> act = static_pointer_cast<const Action>(
        make_shared<const StringAction>("right"));
    shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(
        make_shared<const IntPairState>(1,0));
    
    // Make child
    shared_ptr<TestThtsCNode> r_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> r_node = static_pointer_cast<TestThtsDNode>(r_cnode->create_child_node_itfc(obsv));

    // d action obsv
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

    // rd child
    shared_ptr<TestThtsCNode> rd_cnode = static_pointer_cast<TestThtsCNode>(r_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> rd_node = static_pointer_cast<TestThtsDNode>(rd_cnode->create_child_node_itfc(obsv));

    // d action obsv (from root node)
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(0,1));

    // d child
    shared_ptr<TestThtsCNode> d_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> d_node = static_pointer_cast<TestThtsDNode>(d_cnode->create_child_node_itfc(obsv));

    // r action obsv
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("right"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

    // rd child
    shared_ptr<TestThtsCNode> dr_cnode = static_pointer_cast<TestThtsCNode>(d_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> dr_node = static_pointer_cast<TestThtsDNode>(dr_cnode->create_child_node_itfc(obsv));

    // visit each of the nodes created
    ThtsEnvContext ctx;
    r_cnode->visit_itfc(ctx);
    r_node->visit_itfc(ctx);
    rd_cnode->visit_itfc(ctx);
    rd_node->visit_itfc(ctx);
    d_cnode->visit_itfc(ctx);
    d_node->visit_itfc(ctx);
    dr_cnode->visit_itfc(ctx);
    dr_node->visit_itfc(ctx);

    // Assert that each child has been visited once (other than the root node!)
    EXPECT_EQ(root_node->get_num_visits(), 0);
    EXPECT_EQ(r_cnode->get_num_visits(), 1);
    EXPECT_EQ(r_node->get_num_visits(), 1);
    EXPECT_EQ(rd_cnode->get_num_visits(), 1);
    EXPECT_EQ(rd_node->get_num_visits(), 1);
    EXPECT_EQ(d_cnode->get_num_visits(), 1);
    EXPECT_EQ(d_node->get_num_visits(), 1);
    EXPECT_EQ(dr_cnode->get_num_visits(), 1);
    EXPECT_EQ(dr_node->get_num_visits(), 1);

    // Assert correct decision timesteps and decision depths
    EXPECT_EQ(root_node->get_decision_depth(), mock_decision_depth);
    EXPECT_EQ(r_cnode->get_decision_depth(), mock_decision_depth+0);
    EXPECT_EQ(r_node->get_decision_depth(), mock_decision_depth+1);
    EXPECT_EQ(rd_cnode->get_decision_depth(), mock_decision_depth+1);
    EXPECT_EQ(rd_node->get_decision_depth(), mock_decision_depth+2);
    EXPECT_EQ(d_cnode->get_decision_depth(), mock_decision_depth+0);
    EXPECT_EQ(d_node->get_decision_depth(), mock_decision_depth+1);
    EXPECT_EQ(dr_cnode->get_decision_depth(), mock_decision_depth+1);
    EXPECT_EQ(dr_node->get_decision_depth(), mock_decision_depth+2);

    EXPECT_EQ(root_node->get_decision_timestep(), mock_decision_timestep);
    EXPECT_EQ(r_cnode->get_decision_timestep(), mock_decision_timestep+0);
    EXPECT_EQ(r_node->get_decision_timestep(), mock_decision_timestep+1);
    EXPECT_EQ(rd_cnode->get_decision_timestep(), mock_decision_timestep+1);
    EXPECT_EQ(rd_node->get_decision_timestep(), mock_decision_timestep+2);
    EXPECT_EQ(d_cnode->get_decision_timestep(), mock_decision_timestep+0);
    EXPECT_EQ(d_node->get_decision_timestep(), mock_decision_timestep+1);
    EXPECT_EQ(dr_cnode->get_decision_timestep(), mock_decision_timestep+1);
    EXPECT_EQ(dr_node->get_decision_timestep(), mock_decision_timestep+2);
}

/**
 * Repeat the 'test_normal_usage' test, but this time testing the transposition table.
 * 
 * This is a copy and paste of the previous  this time we set 'mock_manager.use_transposition_table' to true.
 */
TEST(ThtsNode_CreateChild, test_transposition_table)
{   
    // Mock manager
    shared_ptr<MockThtsManager> mock_manager_ptr = make_shared<MockThtsManager>();
    MockThtsManager& mock_manager = *mock_manager_ptr;
    mock_manager.use_transposition_table = true;
    shared_ptr<ThtsManager> manager_ptr = static_pointer_cast<ThtsManager>(mock_manager_ptr);
    EXPECT_CALL(mock_manager, get_rand_int)
        .Times(0);
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(0);
    
    // Mock decision depth and decision timestep
    int mock_decision_depth = 0;
    int mock_decision_timestep = 21;

    //Make env and root node
    shared_ptr<ThtsEnv> thts_env = static_pointer_cast<ThtsEnv>(make_shared<TestThtsEnv>(2));
    shared_ptr<TestThtsDNode> root_node = make_shared<TestThtsDNode>(
        manager_ptr,
        thts_env,
        thts_env->get_initial_state_itfc(),
        mock_decision_depth,
        mock_decision_timestep);

    // Actions, observation pair for making a (r) child
    shared_ptr<const Action> act = static_pointer_cast<const Action>(
        make_shared<const StringAction>("right"));
    shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(
        make_shared<const IntPairState>(1,0));
    
    // Make child
    shared_ptr<TestThtsCNode> r_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> r_node = static_pointer_cast<TestThtsDNode>(r_cnode->create_child_node_itfc(obsv));

    // d action obsv
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

    // rd child
    shared_ptr<TestThtsCNode> rd_cnode = static_pointer_cast<TestThtsCNode>(r_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> rd_node = static_pointer_cast<TestThtsDNode>(rd_cnode->create_child_node_itfc(obsv));

    // d action obsv (from root node)
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(0,1));

    // d child
    shared_ptr<TestThtsCNode> d_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> d_node = static_pointer_cast<TestThtsDNode>(d_cnode->create_child_node_itfc(obsv));

    // r action obsv
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("right"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

    // rd child
    shared_ptr<TestThtsCNode> dr_cnode = static_pointer_cast<TestThtsCNode>(d_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> dr_node = static_pointer_cast<TestThtsDNode>(dr_cnode->create_child_node_itfc(obsv));

    // visit each of the nodes created
    ThtsEnvContext ctx;
    r_cnode->visit_itfc(ctx);
    r_node->visit_itfc(ctx);
    rd_cnode->visit_itfc(ctx);
    rd_node->visit_itfc(ctx);
    d_cnode->visit_itfc(ctx);
    d_node->visit_itfc(ctx);
    dr_cnode->visit_itfc(ctx);
    dr_node->visit_itfc(ctx);

    // Assert that each child has been visited once (other than the root node!)
    EXPECT_EQ(root_node->get_num_visits(), 0);
    EXPECT_EQ(r_cnode->get_num_visits(), 1);
    EXPECT_EQ(r_node->get_num_visits(), 1);
    EXPECT_EQ(rd_cnode->get_num_visits(), 1);
    EXPECT_EQ(rd_node->get_num_visits(), 2);
    EXPECT_EQ(d_cnode->get_num_visits(), 1);
    EXPECT_EQ(d_node->get_num_visits(), 1);
    EXPECT_EQ(dr_cnode->get_num_visits(), 1);
    EXPECT_EQ(dr_node->get_num_visits(), 2);

    // Assert correct decision timesteps and decision depths
    EXPECT_EQ(root_node->get_decision_depth(), mock_decision_depth);
    EXPECT_EQ(r_cnode->get_decision_depth(), mock_decision_depth+0);
    EXPECT_EQ(r_node->get_decision_depth(), mock_decision_depth+1);
    EXPECT_EQ(rd_cnode->get_decision_depth(), mock_decision_depth+1);
    EXPECT_EQ(rd_node->get_decision_depth(), mock_decision_depth+2);
    EXPECT_EQ(d_cnode->get_decision_depth(), mock_decision_depth+0);
    EXPECT_EQ(d_node->get_decision_depth(), mock_decision_depth+1);
    EXPECT_EQ(dr_cnode->get_decision_depth(), mock_decision_depth+1);
    EXPECT_EQ(dr_node->get_decision_depth(), mock_decision_depth+2);

    EXPECT_EQ(root_node->get_decision_timestep(), mock_decision_timestep);
    EXPECT_EQ(r_cnode->get_decision_timestep(), mock_decision_timestep+0);
    EXPECT_EQ(r_node->get_decision_timestep(), mock_decision_timestep+1);
    EXPECT_EQ(rd_cnode->get_decision_timestep(), mock_decision_timestep+1);
    EXPECT_EQ(rd_node->get_decision_timestep(), mock_decision_timestep+2);
    EXPECT_EQ(d_cnode->get_decision_timestep(), mock_decision_timestep+0);
    EXPECT_EQ(d_node->get_decision_timestep(), mock_decision_timestep+1);
    EXPECT_EQ(dr_cnode->get_decision_timestep(), mock_decision_timestep+1);
    EXPECT_EQ(dr_node->get_decision_timestep(), mock_decision_timestep+2);
}



/**
 * Runs same as ThtsNode_CreateChild.test_normal_usage, but checks pretty print at end instead
 */
TEST(ThtsNode_PrettyPrint, test_no_transposition)
{   
    string pretty_print_expected_string =
        "D(vl=0.0,#v=0)[\n"
        "|	\"down\"->C(vl=0.0,#v=1)[\n"
        "|	|	{(0,1)}->D(vl=0.0,#v=1)[\n"
        "|	|	|	\"right\"->C(vl=0.0,#v=1)[\n"
        "|	|	|	|	{(1,1)}->D(vl=0.0,#v=1)[\n"
        "|	|	|	|	],\n"
        "|	|	|	],\n"
        "|	|	],\n"
        "|	],\n"
        "|	\"right\"->C(vl=0.0,#v=1)[\n"
        "|	|	{(1,0)}->D(vl=0.0,#v=1)[\n"
        "|	|	|	\"down\"->C(vl=0.0,#v=1)[\n"
        "|	|	|	|	{(1,1)}->D(vl=0.0,#v=1)[\n"
        "|	|	|	|	],\n"
        "|	|	|	],\n"
        "|	|	],\n"
        "|	],\n"
        "],";

    // Mock manager
    shared_ptr<MockThtsManager> mock_manager_ptr = make_shared<MockThtsManager>();
    MockThtsManager& mock_manager = *mock_manager_ptr;
    shared_ptr<ThtsManager> manager_ptr = static_pointer_cast<ThtsManager>(mock_manager_ptr);
    EXPECT_CALL(mock_manager, get_rand_int)
        .Times(0);
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(0);
    
    // Mock decision depth and decision timestep
    int mock_decision_depth = 0;
    int mock_decision_timestep = 21;

    //Make env and root node
    shared_ptr<ThtsEnv> thts_env = static_pointer_cast<ThtsEnv>(make_shared<TestThtsEnv>(2));
    shared_ptr<TestThtsDNode> root_node = make_shared<TestThtsDNode>(
        manager_ptr,
        thts_env,
        thts_env->get_initial_state_itfc(),
        mock_decision_depth,
        mock_decision_timestep);

    // Actions, observation pair for making a (r) child
    shared_ptr<const Action> act = static_pointer_cast<const Action>(
        make_shared<const StringAction>("right"));
    shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(
        make_shared<const IntPairState>(1,0));
    
    // Make child
    shared_ptr<TestThtsCNode> r_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> r_node = static_pointer_cast<TestThtsDNode>(r_cnode->create_child_node_itfc(obsv));

    // d action obsv
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

    // rd child
    shared_ptr<TestThtsCNode> rd_cnode = static_pointer_cast<TestThtsCNode>(r_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> rd_node = static_pointer_cast<TestThtsDNode>(rd_cnode->create_child_node_itfc(obsv));

    // d action obsv (from root node)
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(0,1));

    // d child
    shared_ptr<TestThtsCNode> d_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> d_node = static_pointer_cast<TestThtsDNode>(d_cnode->create_child_node_itfc(obsv));

    // r action obsv
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("right"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

    // rd child
    shared_ptr<TestThtsCNode> dr_cnode = static_pointer_cast<TestThtsCNode>(d_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> dr_node = static_pointer_cast<TestThtsDNode>(dr_cnode->create_child_node_itfc(obsv));

    // visit each of the nodes created
    ThtsEnvContext ctx;
    r_cnode->visit_itfc(ctx);
    r_node->visit_itfc(ctx);
    rd_cnode->visit_itfc(ctx);
    rd_node->visit_itfc(ctx);
    d_cnode->visit_itfc(ctx);
    d_node->visit_itfc(ctx);
    dr_cnode->visit_itfc(ctx);
    dr_node->visit_itfc(ctx);

    // Assert that each child has been visited once (other than the root node!)
    EXPECT_THAT(root_node->get_pretty_print_string(10), StrEq(pretty_print_expected_string));
}

/**
 * Same as ThtsNode_PrettyPrint.test_no_transposition, but testing pretty print with 
 */
TEST(ThtsNode_PrettyPrint, test_transposition_table)
{   
    string pretty_print_expected_string =
        "D(vl=0.0,#v=0)[\n"
        "|	\"down\"->C(vl=0.0,#v=1)[\n"
        "|	|	{(0,1)}->D(vl=0.0,#v=1)[\n"
        "|	|	|	\"right\"->C(vl=0.0,#v=1)[\n"
        "|	|	|	|	{(1,1)}->D(vl=0.0,#v=2)[\n"
        "|	|	|	|	],\n"
        "|	|	|	],\n"
        "|	|	],\n"
        "|	],\n"
        "|	\"right\"->C(vl=0.0,#v=1)[\n"
        "|	|	{(1,0)}->D(vl=0.0,#v=1)[\n"
        "|	|	|	\"down\"->C(vl=0.0,#v=1)[\n"
        "|	|	|	|	{(1,1)}->D(vl=0.0,#v=2)[\n"
        "|	|	|	|	],\n"
        "|	|	|	],\n"
        "|	|	],\n"
        "|	],\n"
        "],";

    // Mock manager
    shared_ptr<MockThtsManager> mock_manager_ptr = make_shared<MockThtsManager>();
    MockThtsManager& mock_manager = *mock_manager_ptr;
    mock_manager.use_transposition_table = true;
    shared_ptr<ThtsManager> manager_ptr = static_pointer_cast<ThtsManager>(mock_manager_ptr);
    EXPECT_CALL(mock_manager, get_rand_int)
        .Times(0);
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(0);
    
    // Mock decision depth and decision timestep
    int mock_decision_depth = 0;
    int mock_decision_timestep = 21;

    //Make env and root node
    shared_ptr<ThtsEnv> thts_env = static_pointer_cast<ThtsEnv>(make_shared<TestThtsEnv>(2));
    shared_ptr<TestThtsDNode> root_node = make_shared<TestThtsDNode>(
        manager_ptr,
        thts_env,
        thts_env->get_initial_state_itfc(),
        mock_decision_depth,
        mock_decision_timestep);

    // Actions, observation pair for making a (r) child
    shared_ptr<const Action> act = static_pointer_cast<const Action>(
        make_shared<const StringAction>("right"));
    shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(
        make_shared<const IntPairState>(1,0));
    
    // Make child
    shared_ptr<TestThtsCNode> r_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> r_node = static_pointer_cast<TestThtsDNode>(r_cnode->create_child_node_itfc(obsv));

    // d action obsv
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

    // rd child
    shared_ptr<TestThtsCNode> rd_cnode = static_pointer_cast<TestThtsCNode>(r_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> rd_node = static_pointer_cast<TestThtsDNode>(rd_cnode->create_child_node_itfc(obsv));

    // d action obsv (from root node)
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("down"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(0,1));

    // d child
    shared_ptr<TestThtsCNode> d_cnode = static_pointer_cast<TestThtsCNode>(root_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> d_node = static_pointer_cast<TestThtsDNode>(d_cnode->create_child_node_itfc(obsv));

    // r action obsv
    act = static_pointer_cast<const Action>(make_shared<const StringAction>("right"));
    obsv = static_pointer_cast<const Observation>(make_shared<const IntPairState>(1,1));

    // rd child
    shared_ptr<TestThtsCNode> dr_cnode = static_pointer_cast<TestThtsCNode>(d_node->create_child_node_itfc(act));
    shared_ptr<TestThtsDNode> dr_node = static_pointer_cast<TestThtsDNode>(dr_cnode->create_child_node_itfc(obsv));

    // visit each of the nodes created
    ThtsEnvContext ctx;
    r_cnode->visit_itfc(ctx);
    r_node->visit_itfc(ctx);
    rd_cnode->visit_itfc(ctx);
    rd_node->visit_itfc(ctx);
    d_cnode->visit_itfc(ctx);
    d_node->visit_itfc(ctx);
    dr_cnode->visit_itfc(ctx);
    dr_node->visit_itfc(ctx);

    // Assert that each child has been visited once (other than the root node!)
    EXPECT_THAT(root_node->get_pretty_print_string(10), StrEq(pretty_print_expected_string));
}

