#include "test/algorithms/test_puct_nodes.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "algorithms/ments/ments_chance_node.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/ments/ments_manager.h"

// includes
#include "test/test_thts_env.h"
#include "thts.h"

#include <iostream>
#include <unordered_map>


using namespace std;
using namespace thts;
using namespace thts::test;

// actions (for 'EXPECT_CALL')
using ::testing::Return;

// matchers (for 'EXPECT_CALL')

/**
 * Reminder to eventually write unit tests
 */
TEST(Ments_UnitTests, reminder_to_do_at_some_point) {
    FAIL();
}



/**
 * Actually run the full whack on a simple env and test it all works
 * 
 * Prints some fun things out for if we want to read
 */
void run_ments_integration_test(
    int env_size, 
    int num_threads, 
    int num_trials, 
    double stay_prob=0.0, 
    int print_tree_depth=0, 
    double temp=1.0, 
    bool use_temp_decay=false,
    bool use_max_heap=false) 
{
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();

    shared_ptr<ThtsEnv> grid_env = make_shared<TestThtsEnv>(env_size, stay_prob);
    MentsManagerArgs manager_args(grid_env);
    manager_args.seed = 60415;
    manager_args.max_depth = env_size * 4;
    manager_args.mcts_mode = false;
    manager_args.temp = temp;
    manager_args.temp_decay_fn = use_temp_decay ? decayed_temp_inv_sqrt : nullptr;
    manager_args.use_max_heap = use_max_heap;
    shared_ptr<MentsManager> manager = make_shared<MentsManager>(manager_args);
    shared_ptr<MentsDNode> root_node = make_shared<MentsDNode>(manager, grid_env->get_initial_state_itfc(), 0, 0);
    ThtsPool uct_pool(manager, root_node, num_threads);
    uct_pool.run_trials(num_trials);

    if (stay_prob == 0.0) {
        // TODO add asserts
    }

    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    cout << "MENTS with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0){
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    } else {
        cout << endl;
    }
}

TEST(Ments_IntegrationTest, easy_grid_world) {
    run_ments_integration_test(1, 1, 10000, 0.0, 2, 1.0, false, false);
}

TEST(Ments_IntegrationTest, easy_grid_world_multithreaded) {
    run_ments_integration_test(2, 4, 10000, 0.0, 1, 0.5, false, false);
}

TEST(Ments_IntegrationTest, easy_grid_world_stochastic) {
    run_ments_integration_test(1, 1, 10000, 0.1, 2, 1.0, false, false);
}

TEST(Ments_IntegrationTest, easy_grid_world_stochastic_multithreaded) {
    run_ments_integration_test(2, 4, 10000, 0.1, 1, 0.5, false, false);
}



TEST(Ments_WithMaxHeap_IntegrationTest, easy_grid_world) {
    run_ments_integration_test(1, 1, 10000, 0.0, 2, 1.0, false, true);
}

TEST(Ments_WithMaxHeap_IntegrationTest, easy_grid_world_multithreaded) {
    run_ments_integration_test(2, 4, 10000, 0.0, 1, 0.5, false, true);
}

TEST(Ments_WithMaxHeap_IntegrationTest, easy_grid_world_stochastic) {
    run_ments_integration_test(1, 1, 10000, 0.1, 2, 1.0, false, true);
}

TEST(Ments_WithMaxHeap_IntegrationTest, easy_grid_world_stochastic_multithreaded) {
    run_ments_integration_test(2, 4, 10000, 0.1, 1, 0.5, false, true);
}



TEST(Ments_WithTempDecay_IntegrationTest, easy_grid_world) {
    run_ments_integration_test(1, 1, 10000, 0.0, 2, 1.0, true, false);
}

TEST(Ments_WithTempDecay_IntegrationTest, easy_grid_world_multithreaded) {
    run_ments_integration_test(2, 4, 10000, 0.0, 1, 0.5, true, false);
}

TEST(Ments_WithTempDecay_IntegrationTest, easy_grid_world_stochastic) {
    run_ments_integration_test(1, 1, 10000, 0.1, 2, 1.0, true, false);
}

TEST(Ments_WithTempDecay_IntegrationTest, easy_grid_world_stochastic_multithreaded) {
    run_ments_integration_test(2, 4, 10000, 0.1, 1, 0.5, true, false);
}



/**
 * Also run full whack on a simple game to check that the opponent logic all works
 */
void run_ments_game_integration_test(
    int env_size, 
    int num_trials, 
    int print_tree_depth=0, 
    int decision_timestep=0, 
    bool use_temp_decay=false,
    bool use_max_heap=false) 
{
    shared_ptr<ThtsEnv> game_env = make_shared<TestThtsGameEnv>(env_size);
    MentsManagerArgs manager_args(game_env);
    manager_args.seed = 60415;
    manager_args.max_depth = env_size * 4;
    manager_args.mcts_mode = false;
    manager_args.is_two_player_game = true;
    manager_args.temp = 1.0;
    manager_args.temp_decay_fn = use_temp_decay ? decayed_temp_inv_sqrt : nullptr;
    manager_args.use_max_heap = use_max_heap;
    shared_ptr<MentsManager> manager = make_shared<MentsManager>(manager_args);
    shared_ptr<MentsDNode> root_node = make_shared<MentsDNode>(
        manager, game_env->get_initial_state_itfc(), 0, decision_timestep);
    ThtsPool uct_pool(manager, root_node, 1);
    uct_pool.run_trials(num_trials);

    if (print_tree_depth > 0){
        cout << "MENTS with starting decision_timestep of " << decision_timestep << " looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    }
}

TEST(Ments_IntegrationTest, two_player_game_env) {
    run_ments_game_integration_test(3, 10000, 4, 0, false, false);
}

TEST(Ments_IntegrationTest, two_player_game_env_starting_as_opponent) {
    run_ments_game_integration_test(3, 10000, 4, 1, false, false);
}



TEST(Ments_WithMaxHeap_IntegrationTest, two_player_game_env) {
    run_ments_game_integration_test(3, 10000, 4, 0, false, true);
}

TEST(Ments_WithMaxHeap_IntegrationTest, two_player_game_env_starting_as_opponent) {
    run_ments_game_integration_test(3, 10000, 4, 1, false, true);
}



TEST(Ments_WithTempDecay_IntegrationTest, two_player_game_env) {
    run_ments_game_integration_test(3, 10000, 4, 0, true, false);
}

TEST(Ments_WithTempDecay_IntegrationTest, two_player_game_env_starting_as_opponent) {
    run_ments_game_integration_test(3, 10000, 4, 1, true, false);
}