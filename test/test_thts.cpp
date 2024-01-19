#include "test_thts.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "thts.h"

// includes
#include "test_thts_env.h"
#include "test_thts_nodes.h"

using namespace std;
using namespace std::chrono_literals;
using namespace thts;
using namespace thts::test;

// actions (for 'EXPECT_CALL')
using ::testing::Return;

// matchers (for 'EXPECT_CALL')
using ::testing::_;
using ::testing::IsEmpty;
using ::testing::ElementsAre;


/**
 * Test that not providing a root node with the default ThtsPool throws an exception
 */
TEST(ThtsPool_ErrorChecking, check_no_root_node_throws_exception) {
    shared_ptr<ThtsEnv> dummy_env = make_shared<TestThtsEnv>(2);
    shared_ptr<ThtsManager> dummy_manager = make_shared<ThtsManager>(ThtsManagerArgs(dummy_env));
    shared_ptr<const IntPairState> dummy_init_state = ((TestThtsEnv&) *dummy_env).get_initial_state();
    shared_ptr<ThtsDNode> dummy_root_node = make_shared<TestThtsDNode>(dummy_manager,dummy_init_state,0,0);

    EXPECT_ANY_THROW(ThtsPool(nullptr, nullptr));
    EXPECT_ANY_THROW(ThtsPool(dummy_manager, nullptr));
    EXPECT_ANY_THROW(ThtsPool(nullptr, dummy_root_node));
}

/**
 * Test that we can just make and destroy a ThtsPool with multiple threads without a problem
 * And check that default values at construction imply that 'work_left() == false"
 */
TEST(ThtsPool_TestThreadPool, construct_and_destruct_sound) {
    shared_ptr<ThtsEnv> dummy_env = make_shared<TestThtsEnv>(2);
    shared_ptr<ThtsManager> dummy_manager = make_shared<ThtsManager>(ThtsManagerArgs(dummy_env));
    shared_ptr<const IntPairState> dummy_init_state = ((TestThtsEnv&) *dummy_env).get_initial_state();
    shared_ptr<ThtsDNode> dummy_root_node = make_shared<TestThtsDNode>(dummy_manager,dummy_init_state,0,0);
    int num_threads = 4;

    MockThtsPool_PoolTesting* mock_pool = new MockThtsPool_PoolTesting(dummy_manager, dummy_root_node, num_threads);
    EXPECT_FALSE(mock_pool->work_left());
    delete mock_pool;
}

/**
 * Test 'work_left' function works correctly
 */
TEST(ThtsPool_TestThreadPool, check_work_left_function) {
    shared_ptr<ThtsEnv> dummy_env = make_shared<TestThtsEnv>(2);
    shared_ptr<ThtsManager> dummy_manager = make_shared<ThtsManager>(ThtsManagerArgs(dummy_env));
    shared_ptr<const IntPairState> dummy_init_state = ((TestThtsEnv&) *dummy_env).get_initial_state();
    shared_ptr<ThtsDNode> dummy_root_node = make_shared<TestThtsDNode>(dummy_manager,dummy_init_state,0,0);

    MockThtsPool_PoolTesting mock_pool(dummy_manager, dummy_root_node);
    EXPECT_CALL(mock_pool, run_thts_trial)
        .Times(0);

    // time left and trials remaining
    int trials_remaining = 10;
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    chrono::duration<double> max_run_time = chrono::duration<double>(1.0);
    mock_pool.mock_work_left_scenario(trials_remaining, start_time, max_run_time);

    EXPECT_TRUE(mock_pool.work_left());

    // time left and no trials remaining trials remaining
    trials_remaining = 0;
    start_time = chrono::system_clock::now();
    max_run_time = chrono::duration<double>(1.0);
    mock_pool.mock_work_left_scenario(trials_remaining, start_time, max_run_time);

    EXPECT_FALSE(mock_pool.work_left());

    // no time left and trials remaining trials remaining
    trials_remaining = 100;
    start_time = chrono::system_clock::now() - 2s;
    max_run_time = chrono::duration<double>(1.0);
    mock_pool.mock_work_left_scenario(trials_remaining, start_time, max_run_time);

    EXPECT_FALSE(mock_pool.work_left());

    // no time left and negative trials remaining trials remaining
    trials_remaining = -100;
    start_time = chrono::system_clock::now() - 2s;
    max_run_time = chrono::duration<double>(1.0);
    mock_pool.mock_work_left_scenario(trials_remaining, start_time, max_run_time);

    EXPECT_FALSE(mock_pool.work_left());   
}

/**
 * Test that the correct number of trials is run when calling 'run_trials'.
 * 
 * Checks does correct number of trials even with multithreading.
 */
TEST(ThtsPool_TestThreadPool, test_run_trials) {
    int num_trials = 100;

    shared_ptr<ThtsEnv> dummy_env = make_shared<TestThtsEnv>(2);
    ThtsManagerArgs manager_args(dummy_env);
    shared_ptr<ThtsManager> dummy_manager = make_shared<ThtsManager>(manager_args);
    shared_ptr<const IntPairState> dummy_init_state = ((TestThtsEnv&) *dummy_env).get_initial_state();
    shared_ptr<ThtsDNode> dummy_root_node = make_shared<TestThtsDNode>(dummy_manager,dummy_init_state,0,0);
    int num_threads = 4;

    MockThtsPool_PoolTesting mock_pool(dummy_manager, dummy_root_node, num_threads);
    EXPECT_CALL(mock_pool, run_thts_trial)
        .Times(num_trials);

    mock_pool.run_trials(num_trials);

    EXPECT_FALSE(mock_pool.work_left());
}

/**
 * Test that a non-blocking call to run trials is non-blocking. 
 */
TEST(ThtsPool_TestThreadPool, test_run_trials_non_blocking) {
    int num_trials = 4;
    int run_trial_duration_ms = 100;

    shared_ptr<ThtsEnv> dummy_env = make_shared<TestThtsEnv>(2);
    ThtsManagerArgs manager_args(dummy_env);
    shared_ptr<ThtsManager> dummy_manager = make_shared<ThtsManager>(manager_args);
    shared_ptr<const IntPairState> dummy_init_state = ((TestThtsEnv&) *dummy_env).get_initial_state();
    shared_ptr<ThtsDNode> dummy_root_node = make_shared<TestThtsDNode>(dummy_manager,dummy_init_state,0,0);
    int num_threads = 2;

    MockThtsPool_DurationTrialPoolTesting mock_pool(run_trial_duration_ms, dummy_manager, dummy_root_node, num_threads);

    double max_time = 1000.0;
    bool blocking = false;
    mock_pool.run_trials(num_trials, max_time, blocking);

    // check didn't block by grabbing the work_left_lock, and checking that there is still work left to do
    mutex& work_left_lock = mock_pool.get_work_left_lock();
    work_left_lock.lock();
    EXPECT_TRUE(mock_pool.work_left());
    work_left_lock.unlock();

    // joint and then check work left is false again
    mock_pool.join();
    EXPECT_FALSE(mock_pool.work_left());
}

/**
 * Test that the correct number of trials is run when calling 'run_trials'.
 * 
 * Times how long it takes for two trials to be run by 2 threads. If concurrent, then it should be quicker than 
 * running the two trials sequentially.
 * 
 * N.B. hard coded duration check at the end, because chronos is a bit of a pain as a newbie.
 */
TEST(ThtsPool_TestThreadPool, test_trials_run_concurrently) {
    int num_trials = 2;
    int run_trial_duration_ms = 100;

    shared_ptr<ThtsEnv> dummy_env = make_shared<TestThtsEnv>(2);
    ThtsManagerArgs manager_args(dummy_env);
    shared_ptr<ThtsManager> dummy_manager = make_shared<ThtsManager>(manager_args);
    shared_ptr<const IntPairState> dummy_init_state = ((TestThtsEnv&) *dummy_env).get_initial_state();
    shared_ptr<ThtsDNode> dummy_root_node = make_shared<TestThtsDNode>(dummy_manager,dummy_init_state,0,0);
    int num_threads = 2;

    MockThtsPool_DurationTrialPoolTesting mock_pool(run_trial_duration_ms, dummy_manager, dummy_root_node, num_threads);

    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();
    mock_pool.run_trials(num_trials);
    chrono::time_point<chrono::system_clock> end_time = chrono::system_clock::now();
    chrono::duration<double> trials_duration = end_time - start_time;
    
    // check total time taken < total time needed for sequential execution
    EXPECT_LE(trials_duration, 200ms);
}


/**
 * Checks that 'should_continue_selection_phase' returns false when passed a leaf node of node at the max decision 
 * depth. And aditionally returns false when a 'new_decision_node_created_this_trial' is set to true in mcts mode.
 */
TEST(ThtsPool_TestRunTrial, test_should_continue_selection_phase) {
    // Make a pool so can run function, don't actually need any workers anyway
    int dummy_max_depth = 100;
    shared_ptr<ThtsEnv> dummy_env = make_shared<TestThtsEnv>(2);
    ThtsManagerArgs manager_args(dummy_env);
    manager_args.max_depth=dummy_max_depth;
    shared_ptr<ThtsManager> dummy_manager = make_shared<ThtsManager>(manager_args);
    shared_ptr<const IntPairState> dummy_init_state = ((TestThtsEnv&) *dummy_env).get_initial_state();
    shared_ptr<MockThtsDNode> mock_node = make_shared<MockThtsDNode>(dummy_manager,dummy_init_state,0,0);
    shared_ptr<ThtsDNode> dummy_root_node = static_pointer_cast<ThtsDNode>(mock_node);

    int num_threads = 0;
    PublicThtsPool thts_pool(dummy_manager, dummy_root_node, num_threads);

    // mcts mode, is leaf
    shared_ptr<MockThtsDNode> mock_search_node = make_shared<MockThtsDNode>(
        dummy_manager,dummy_init_state,dummy_max_depth-10,0);
    EXPECT_CALL(*mock_search_node, is_sink)
        .Times(1)
        .WillOnce(Return(true));
    EXPECT_FALSE(thts_pool.should_continue_selection_phase(mock_search_node, false));

    // mcts mode, max decision depth
    mock_search_node = make_shared<MockThtsDNode>(dummy_manager,dummy_init_state,dummy_max_depth,0);
    EXPECT_CALL(*mock_search_node, is_sink)
        .Times(1)
        .WillOnce(Return(false));
    EXPECT_FALSE(thts_pool.should_continue_selection_phase(mock_search_node, false));

    // mcts mode, new node made
    mock_search_node = make_shared<MockThtsDNode>(dummy_manager,dummy_init_state,dummy_max_depth-10,0);
    EXPECT_CALL(*mock_search_node, is_sink)
        .Times(1)
        .WillOnce(Return(false));
    EXPECT_FALSE(thts_pool.should_continue_selection_phase(mock_search_node, true));

    // mcts mode, new node made
    mock_search_node = make_shared<MockThtsDNode>(dummy_manager,dummy_init_state,dummy_max_depth-10,0);
    EXPECT_CALL(*mock_search_node, is_sink)
        .Times(1)
        .WillOnce(Return(false));
    EXPECT_TRUE(thts_pool.should_continue_selection_phase(mock_search_node, false));

    // uct mode, is leaf
    dummy_manager->mcts_mode = false;
    mock_search_node = make_shared<MockThtsDNode>(dummy_manager,dummy_init_state,dummy_max_depth-10,0);
    EXPECT_CALL(*mock_search_node, is_sink)
        .Times(1)
        .WillOnce(Return(true));
    EXPECT_FALSE(thts_pool.should_continue_selection_phase(mock_search_node, false));

    // uct mode, max decision depth
    mock_search_node = make_shared<MockThtsDNode>(dummy_manager,dummy_init_state,dummy_max_depth,0);
    EXPECT_CALL(*mock_search_node, is_sink)
        .Times(1)
        .WillOnce(Return(false));
    EXPECT_FALSE(thts_pool.should_continue_selection_phase(mock_search_node, false));

    // uct mode, new node made
    mock_search_node = make_shared<MockThtsDNode>(dummy_manager,dummy_init_state,dummy_max_depth-10,0);
    EXPECT_CALL(*mock_search_node, is_sink)
        .Times(1)
        .WillOnce(Return(false));
    EXPECT_TRUE(thts_pool.should_continue_selection_phase(mock_search_node, true));

    // uct mode, new node made
    mock_search_node = make_shared<MockThtsDNode>(dummy_manager,dummy_init_state,dummy_max_depth-10,0);
    EXPECT_CALL(*mock_search_node, is_sink)
        .Times(1)
        .WillOnce(Return(false));
    EXPECT_TRUE(thts_pool.should_continue_selection_phase(mock_search_node, false));
}

/**
 * Tests normal operation of selection phase
 */
TEST(ThtsPool_TestRunTrial, test_selection_phase) {
    // Make mocks
    shared_ptr<MockTestThtsEnv> mock_env_ptr = make_shared<MockTestThtsEnv>(2);
    MockTestThtsEnv& mock_env = *mock_env_ptr;
    shared_ptr<ThtsEnv> env_ptr = static_pointer_cast<ThtsEnv>(mock_env_ptr);
    ThtsManagerArgs manager_arg(env_ptr);
    shared_ptr<ThtsManager> manager_ptr = make_shared<ThtsManager>(manager_arg);
    // ThtsManager& manager = *manager_ptr;
    shared_ptr<const IntPairState> mock_init_state = mock_env.get_initial_state();
    shared_ptr<MockThtsDNode> mock_root_node_ptr = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    // MockThtsDNode& mock_root_node = *mock_root_node_ptr;
    shared_ptr<ThtsDNode> root_node_ptr = static_pointer_cast<ThtsDNode>(mock_root_node_ptr);

    int num_threads = 0;
    MockPublicThtsPool thts_pool(manager_ptr, root_node_ptr, num_threads);

    // Make a sequence of nodes we will pass through in the trial
    shared_ptr<MockThtsDNode> dnode0 = mock_root_node_ptr;
    shared_ptr<MockThtsCNode> cnode0 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);
    shared_ptr<MockThtsDNode> dnode1 = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    shared_ptr<MockThtsCNode> cnode1 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);
    shared_ptr<MockThtsDNode> dnode2 = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    shared_ptr<MockThtsCNode> cnode2 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);
    shared_ptr<MockThtsDNode> dnode3 = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    shared_ptr<MockThtsCNode> cnode3 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);
    shared_ptr<MockThtsDNode> dnode4 = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    shared_ptr<MockThtsCNode> cnode4 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);

    shared_ptr<ThtsDNode> dnode0_itfc = static_pointer_cast<ThtsDNode>(dnode0);
    shared_ptr<ThtsCNode> cnode0_itfc = static_pointer_cast<ThtsCNode>(cnode0);
    shared_ptr<ThtsDNode> dnode1_itfc = static_pointer_cast<ThtsDNode>(dnode1);
    shared_ptr<ThtsCNode> cnode1_itfc = static_pointer_cast<ThtsCNode>(cnode1);
    shared_ptr<ThtsDNode> dnode2_itfc = static_pointer_cast<ThtsDNode>(dnode2);
    shared_ptr<ThtsCNode> cnode2_itfc = static_pointer_cast<ThtsCNode>(cnode2);
    shared_ptr<ThtsDNode> dnode3_itfc = static_pointer_cast<ThtsDNode>(dnode3);
    shared_ptr<ThtsCNode> cnode3_itfc = static_pointer_cast<ThtsCNode>(cnode3);
    shared_ptr<ThtsDNode> dnode4_itfc = static_pointer_cast<ThtsDNode>(dnode4);
    shared_ptr<ThtsCNode> cnode4_itfc = static_pointer_cast<ThtsCNode>(cnode4);

    shared_ptr<const Action> dummy_action = make_shared<IntAction>(0);
    shared_ptr<const Observation> dummy_observation = make_shared<IntState>(0);

    // Trial will run from dnode0 -> dnode4
    EXPECT_CALL(thts_pool, should_continue_selection_phase)
        .Times(5)
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(false));
    
    EXPECT_CALL(mock_env, get_reward_itfc)
        .Times(4)
        .WillOnce(Return(-2.0))
        .WillOnce(Return(-4.0))
        .WillOnce(Return(-8.0))
        .WillOnce(Return(-16.0));

    EXPECT_CALL(*dnode0, visit_itfc)
        .Times(1);
    EXPECT_CALL(*dnode0, select_action_itfc)
        .Times(1)
        .WillRepeatedly(Return(dummy_action));
    EXPECT_CALL(*dnode0, backup_itfc) 
        .Times(0);
    EXPECT_CALL(*dnode0, get_child_node_itfc(dummy_action))
        .Times(1)
        .WillRepeatedly(Return(cnode0));
    dnode0->set_heuristic_value(1.5);

    EXPECT_CALL(*cnode0, visit_itfc)
        .Times(1);
    EXPECT_CALL(*cnode0, sample_observation_itfc)
        .Times(1)
        .WillRepeatedly(Return(dummy_observation));
    EXPECT_CALL(*cnode0, backup_itfc)
        .Times(0);
    EXPECT_CALL(*cnode0, get_num_children)
        .Times(2)
        .WillRepeatedly(Return(5));
    EXPECT_CALL(*cnode0, get_child_node_itfc(dummy_observation))
        .Times(1)
        .WillRepeatedly(Return(dnode1));

    EXPECT_CALL(*dnode1, visit_itfc)
        .Times(1);
    EXPECT_CALL(*dnode1, select_action_itfc)
        .Times(1)
        .WillRepeatedly(Return(dummy_action));
    EXPECT_CALL(*dnode1, backup_itfc) 
        .Times(0);
    EXPECT_CALL(*dnode1, get_child_node_itfc(dummy_action))
        .Times(1)
        .WillRepeatedly(Return(cnode1));
    dnode1->set_heuristic_value(1.5);

    EXPECT_CALL(*cnode1, visit_itfc)
        .Times(1);
    EXPECT_CALL(*cnode1, sample_observation_itfc)
        .Times(1)
        .WillRepeatedly(Return(dummy_observation));
    EXPECT_CALL(*cnode1, backup_itfc)
        .Times(0);
    EXPECT_CALL(*cnode1, get_num_children)
        .Times(2)
        .WillRepeatedly(Return(2));
    EXPECT_CALL(*cnode1, get_child_node_itfc(dummy_observation))
        .Times(1)
        .WillRepeatedly(Return(dnode2));

    EXPECT_CALL(*dnode2, visit_itfc)
        .Times(1);
    EXPECT_CALL(*dnode2, select_action_itfc)
        .Times(1)
        .WillRepeatedly(Return(dummy_action));
    EXPECT_CALL(*dnode2, backup_itfc) 
        .Times(0);
    EXPECT_CALL(*dnode2, get_child_node_itfc(dummy_action))
        .Times(1)
        .WillRepeatedly(Return(cnode2));
    dnode2->set_heuristic_value(1.5);

    EXPECT_CALL(*cnode2, visit_itfc)
        .Times(1);
    EXPECT_CALL(*cnode2, sample_observation_itfc)
        .Times(1)
        .WillRepeatedly(Return(dummy_observation));
    EXPECT_CALL(*cnode2, backup_itfc)
        .Times(0);
    EXPECT_CALL(*cnode2, get_num_children)
        .Times(2)
        .WillRepeatedly(Return(100));
    EXPECT_CALL(*cnode2, get_child_node_itfc(dummy_observation))
        .Times(1)
        .WillRepeatedly(Return(dnode3));

    EXPECT_CALL(*dnode3, visit_itfc)
        .Times(1);
    EXPECT_CALL(*dnode3, select_action_itfc)
        .Times(1)
        .WillRepeatedly(Return(dummy_action));
    EXPECT_CALL(*dnode3, backup_itfc) 
        .Times(0);
    EXPECT_CALL(*dnode3, get_child_node_itfc(dummy_action))
        .Times(1)
        .WillRepeatedly(Return(cnode3));
    dnode3->set_heuristic_value(1.5);

    EXPECT_CALL(*cnode3, visit_itfc)
        .Times(1);
    EXPECT_CALL(*cnode3, sample_observation_itfc)
        .Times(1)
        .WillRepeatedly(Return(dummy_observation));
    EXPECT_CALL(*cnode3, backup_itfc)
        .Times(0);
    EXPECT_CALL(*cnode3, get_num_children)
        .Times(2)
        .WillOnce(Return(9))
        .WillOnce(Return(10));
    EXPECT_CALL(*cnode3, get_child_node_itfc(dummy_observation))
        .Times(1)
        .WillRepeatedly(Return(dnode4));

    EXPECT_CALL(*dnode4, visit_itfc)
        .Times(1);
    EXPECT_CALL(*dnode4, select_action_itfc)
        .Times(0);
    EXPECT_CALL(*dnode4, backup_itfc) 
        .Times(0);
    EXPECT_CALL(*dnode4, get_child_node_itfc(dummy_action))
        .Times(0);
    dnode4->set_heuristic_value(1.0);

    EXPECT_CALL(*cnode4, visit_itfc)
        .Times(0);
    EXPECT_CALL(*cnode4, sample_observation_itfc)
        .Times(0);
    EXPECT_CALL(*cnode4, backup_itfc)
        .Times(0);
    EXPECT_CALL(*cnode4, get_num_children)
        .Times(0);
    EXPECT_CALL(*cnode4, get_child_node_itfc(dummy_observation))
        .Times(0);
        
    // Run a selection phase
    vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>> nodes_to_backup;
    vector<double> rewards;
    shared_ptr<ThtsEnvContext> context = env_ptr->sample_context_itfc(0,*manager_ptr);

    thts_pool.run_selection_phase(nodes_to_backup, rewards, *context, 0);

    // Check lengths, rewards should include the heuristic value of dnode4 too
    EXPECT_EQ(nodes_to_backup.size(), 4u);
    EXPECT_EQ(rewards.size(), 5u);   

    // Check contencts of vectors
    EXPECT_EQ(nodes_to_backup[0].first, dnode0_itfc);
    EXPECT_EQ(nodes_to_backup[0].second, cnode0_itfc);
    EXPECT_EQ(nodes_to_backup[1].first, dnode1_itfc);
    EXPECT_EQ(nodes_to_backup[1].second, cnode1_itfc);
    EXPECT_EQ(nodes_to_backup[2].first, dnode2_itfc);
    EXPECT_EQ(nodes_to_backup[2].second, cnode2_itfc);
    EXPECT_EQ(nodes_to_backup[3].first, dnode3_itfc);
    EXPECT_EQ(nodes_to_backup[3].second, cnode3_itfc);

    EXPECT_EQ(rewards[0], -2.0);
    EXPECT_EQ(rewards[1], -4.0);
    EXPECT_EQ(rewards[2], -8.0);
    EXPECT_EQ(rewards[3], -16.0);
    EXPECT_EQ(rewards[4], 1.0);
}

/**
 * Tests normal operation of backup phase
 */
TEST(ThtsPool_TestRunTrial, test_backup_phase) {
    // Make mocks
    shared_ptr<MockTestThtsEnv> mock_env_ptr = make_shared<MockTestThtsEnv>(2);
    MockTestThtsEnv& mock_env = *mock_env_ptr;
    shared_ptr<ThtsEnv> env_ptr = static_pointer_cast<ThtsEnv>(mock_env_ptr);
    ThtsManagerArgs manager_args(env_ptr);
    shared_ptr<ThtsManager> manager_ptr = make_shared<ThtsManager>(manager_args);
    // ThtsManager& manager = *manager_ptr;
    shared_ptr<const IntPairState> mock_init_state = mock_env.get_initial_state();
    shared_ptr<MockThtsDNode> mock_root_node_ptr = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    // MockThtsDNode& mock_root_node = *mock_root_node_ptr;
    shared_ptr<ThtsDNode> root_node_ptr = static_pointer_cast<ThtsDNode>(mock_root_node_ptr);

    int num_threads = 0;
    MockPublicThtsPool thts_pool(manager_ptr, root_node_ptr, num_threads);

    // Make a sequence of nodes we will pass through in the trial
    shared_ptr<MockThtsDNode> dnode0 = mock_root_node_ptr;
    shared_ptr<MockThtsCNode> cnode0 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);
    shared_ptr<MockThtsDNode> dnode1 = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    shared_ptr<MockThtsCNode> cnode1 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);
    shared_ptr<MockThtsDNode> dnode2 = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    shared_ptr<MockThtsCNode> cnode2 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);
    shared_ptr<MockThtsDNode> dnode3 = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    shared_ptr<MockThtsCNode> cnode3 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);
    shared_ptr<MockThtsDNode> dnode4 = make_shared<MockThtsDNode>(manager_ptr,mock_init_state,0,0);
    shared_ptr<MockThtsCNode> cnode4 = make_shared<MockThtsCNode>(manager_ptr,mock_init_state,nullptr,0,0);

    shared_ptr<ThtsDNode> dnode0_itfc = static_pointer_cast<ThtsDNode>(dnode0);
    shared_ptr<ThtsCNode> cnode0_itfc = static_pointer_cast<ThtsCNode>(cnode0);
    shared_ptr<ThtsDNode> dnode1_itfc = static_pointer_cast<ThtsDNode>(dnode1);
    shared_ptr<ThtsCNode> cnode1_itfc = static_pointer_cast<ThtsCNode>(cnode1);
    shared_ptr<ThtsDNode> dnode2_itfc = static_pointer_cast<ThtsDNode>(dnode2);
    shared_ptr<ThtsCNode> cnode2_itfc = static_pointer_cast<ThtsCNode>(cnode2);
    shared_ptr<ThtsDNode> dnode3_itfc = static_pointer_cast<ThtsDNode>(dnode3);
    shared_ptr<ThtsCNode> cnode3_itfc = static_pointer_cast<ThtsCNode>(cnode3);
    shared_ptr<ThtsDNode> dnode4_itfc = static_pointer_cast<ThtsDNode>(dnode4);
    shared_ptr<ThtsCNode> cnode4_itfc = static_pointer_cast<ThtsCNode>(cnode4);

    shared_ptr<const Action> dummy_action = make_shared<IntAction>(0);
    shared_ptr<const Observation> dummy_observation = make_shared<IntState>(0);

    // Trial ran from dnode0 -> dnode4
    EXPECT_CALL(*dnode0, visit_itfc)
        .Times(0);
    EXPECT_CALL(*dnode0, select_action_itfc)
        .Times(0);
    EXPECT_CALL(*dnode0, backup_itfc(IsEmpty(),ElementsAre(1.0,-16.0,-8.0,-4.0,-2.0),-29.0,-29.0,_)) 
        .Times(1);
    EXPECT_CALL(*dnode0, get_child_node_itfc)
        .Times(0);

    EXPECT_CALL(*cnode0, visit_itfc)
        .Times(0);
    EXPECT_CALL(*cnode0, sample_observation_itfc)
        .Times(0);
    EXPECT_CALL(*cnode0, backup_itfc(IsEmpty(),ElementsAre(1.0,-16.0,-8.0,-4.0,-2.0),-29.0,-29.0,_)) 
        .Times(1);
    EXPECT_CALL(*cnode0, get_num_children)
        .Times(0);
    EXPECT_CALL(*cnode0, get_child_node_itfc)
        .Times(0);

    EXPECT_CALL(*dnode1, visit_itfc)
        .Times(0);
    EXPECT_CALL(*dnode1, select_action_itfc)
        .Times(0);
    EXPECT_CALL(*dnode1, backup_itfc(ElementsAre(-2.0),ElementsAre(1.0,-16.0,-8.0,-4.0),-27.0,-29.0,_)) 
        .Times(1);
    EXPECT_CALL(*dnode1, get_child_node_itfc(dummy_action))
        .Times(0);

    EXPECT_CALL(*cnode1, visit_itfc)
        .Times(0);
    EXPECT_CALL(*cnode1, sample_observation_itfc)
        .Times(0);
    EXPECT_CALL(*cnode1, backup_itfc(ElementsAre(-2.0),ElementsAre(1.0,-16.0,-8.0,-4.0),-27.0,-29.0,_)) 
        .Times(1);
    EXPECT_CALL(*cnode1, get_num_children)
        .Times(0);
    EXPECT_CALL(*cnode1, get_child_node_itfc(dummy_observation))
        .Times(0);

    EXPECT_CALL(*dnode2, visit_itfc)
        .Times(0);
    EXPECT_CALL(*dnode2, select_action_itfc)
        .Times(0);
    EXPECT_CALL(*dnode2, backup_itfc(ElementsAre(-2.0,-4.0),ElementsAre(1.0,-16.0,-8.0),-23.0,-29.0,_)) 
        .Times(1);
    EXPECT_CALL(*dnode2, get_child_node_itfc(dummy_action))
        .Times(0);

    EXPECT_CALL(*cnode2, visit_itfc)
        .Times(0);
    EXPECT_CALL(*cnode2, sample_observation_itfc)
        .Times(0);
    EXPECT_CALL(*cnode2, backup_itfc(ElementsAre(-2.0,-4.0),ElementsAre(1.0,-16.0,-8.0),-23.0,-29.0,_)) 
        .Times(1);
    EXPECT_CALL(*cnode2, get_num_children)
        .Times(0);
    EXPECT_CALL(*cnode2, get_child_node_itfc(dummy_observation))
        .Times(0);

    EXPECT_CALL(*dnode3, visit_itfc)
        .Times(0);
    EXPECT_CALL(*dnode3, select_action_itfc)
        .Times(0);
    EXPECT_CALL(*dnode3, backup_itfc(ElementsAre(-2.0,-4.0,-8.0),ElementsAre(1.0,-16.0),-15.0,-29.0,_)) 
        .Times(1);
    EXPECT_CALL(*dnode3, get_child_node_itfc(dummy_action))
        .Times(0);

    EXPECT_CALL(*cnode3, visit_itfc)
        .Times(0);
    EXPECT_CALL(*cnode3, sample_observation_itfc)
        .Times(0);
    EXPECT_CALL(*cnode3, backup_itfc(ElementsAre(-2.0,-4.0,-8.0),ElementsAre(1.0,-16.0),-15.0,-29.0,_)) 
        .Times(1);
    EXPECT_CALL(*cnode3, get_num_children)
        .Times(0);
    EXPECT_CALL(*cnode3, get_child_node_itfc(dummy_observation))
        .Times(0);

    EXPECT_CALL(*dnode4, visit_itfc)
        .Times(0);
    EXPECT_CALL(*dnode4, select_action_itfc)
        .Times(0);
    EXPECT_CALL(*dnode4, backup_itfc) 
        .Times(0);
    EXPECT_CALL(*dnode4, get_child_node_itfc(dummy_action))
        .Times(0);

    EXPECT_CALL(*cnode4, visit_itfc)
        .Times(0);
    EXPECT_CALL(*cnode4, sample_observation_itfc)
        .Times(0);
    EXPECT_CALL(*cnode4, backup_itfc) 
        .Times(0);
    EXPECT_CALL(*cnode4, get_num_children)
        .Times(0);
    EXPECT_CALL(*cnode4, get_child_node_itfc(dummy_observation))
        .Times(0);
        
    // Run a backup phase
    vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>> nodes_to_backup;
    nodes_to_backup.push_back(make_pair(dnode0, cnode0));
    nodes_to_backup.push_back(make_pair(dnode1, cnode1));
    nodes_to_backup.push_back(make_pair(dnode2, cnode2));
    nodes_to_backup.push_back(make_pair(dnode3, cnode3));

    vector<double> rewards = {-2.0, -4.0, -8.0, -16.0, 1.0};
    shared_ptr<ThtsEnvContext> context = env_ptr->sample_context_itfc(0,*manager_ptr);

    thts_pool.run_backup_phase(nodes_to_backup, rewards, *context);
}



TEST(ThtsPool, todo_add_tests_for_sample_context_and_reset_itfc_and_register_thts_context_and_register_thread_id) {
    FAIL();
}