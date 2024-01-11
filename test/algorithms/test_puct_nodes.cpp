#include "test/algorithms/test_puct_nodes.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "algorithms/uct/puct_chance_node.h"
#include "algorithms/uct/puct_decision_node.h"
#include "algorithms/uct/puct_manager.h"

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


/**
 * Reminder to eventually write unit tests
 */
TEST(Puct_UnitTests, reminder_to_do_at_some_point) {
    FAIL();
}



/**
 * Actually run the full whack on a simple env and test it all works
 * 
 * Prints some fun things out for if we want to read
 */
void run_puct_integration_test(int env_size, int num_threads, int num_trials, double stay_prob=0.0, int print_tree_depth=0) {
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();

    shared_ptr<ThtsEnv> grid_env = make_shared<TestThtsEnv>(env_size, stay_prob);
    PuctManagerArgs manager_args(grid_env);
    manager_args.seed = 60415;
    manager_args.max_depth = env_size * 4;
    manager_args.mcts_mode = false;
    shared_ptr<PuctManager> manager = make_shared<PuctManager>(manager_args);
    shared_ptr<PuctDNode> root_node = make_shared<PuctDNode>(
        manager, grid_env->get_initial_state_itfc(), 0, 0);
    ThtsPool thts_pool(manager, root_node, num_threads);
    thts_pool.run_trials(num_trials);

    if (stay_prob == 0.0) {
        // TODO add asserts
    }

    std::chrono::duration<double> dur = chrono::system_clock::now() - start_time;

    cout << "PUCT with " << num_threads << " threads (took " << dur.count() << ")";
    if (print_tree_depth > 0){
        cout << " and looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    } else {
        cout << endl;
    }
}

TEST(Puct_IntegrationTest, easy_grid_world) {
    run_puct_integration_test(1,1,10000,0.0,2);
}

TEST(Puct_IntegrationTest, easy_grid_world_multithreaded) {
    run_puct_integration_test(2,4,10000,0.0,1);
}

TEST(Puct_IntegrationTest, easy_grid_world_stochastic) {
    run_puct_integration_test(1,1,10000,0.1,2);
}

TEST(Puct_IntegrationTest, easy_grid_world_stochastic_multithreaded) {
    run_puct_integration_test(2,4,10000,0.1,1);
}



/**
 * Also run full whack on a simple game to check that the opponent logic all works
 */
void run_puct_game_integration_test(int env_size, int num_trials, int print_tree_depth=0, int decision_timestep=0) {
    shared_ptr<ThtsEnv> game_env = make_shared<TestThtsGameEnv>(env_size);
    PuctManagerArgs manager_args(game_env);
    manager_args.seed = 60415;
    manager_args.max_depth = env_size * 4;
    manager_args.mcts_mode = false;
    manager_args.is_two_player_game = true;
    shared_ptr<PuctManager> manager = make_shared<PuctManager>(manager_args);
    shared_ptr<PuctDNode> root_node = make_shared<PuctDNode>(
        manager, game_env->get_initial_state_itfc(), 0, decision_timestep);
    ThtsPool thts_pool(manager, root_node, 1);
    thts_pool.run_trials(num_trials);

    if (print_tree_depth > 0){
        cout << "PUCT with starting decision_timestep of " << decision_timestep << " looks like:\n";
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    }
}

TEST(Puct_IntegrationTest, two_player_game_env) {
    run_puct_game_integration_test(3, 10000, 4, 0);
}

TEST(Puct_IntegrationTest, two_player_game_env_starting_as_opponent) {
    run_puct_game_integration_test(3, 10000, 4, 1);
}