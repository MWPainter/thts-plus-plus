#include "test/algorithms/test_puct_nodes.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "algorithms/ments_chance_node.h"
#include "algorithms/ments_decision_node.h"
#include "algorithms/ments_manager.h"

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

// matchers (for 'EXPECT_CALL')

#include <iostream>

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
    int env_size, int num_threads, int num_trials, double stay_prob=0.0, int print_tree_depth=0, double temp=1.0) 
{
    chrono::time_point<chrono::system_clock> start_time = chrono::system_clock::now();

    shared_ptr<ThtsEnv> grid_env = make_shared<TestThtsEnv>(env_size, stay_prob);
    shared_ptr<MentsManager> manager = make_shared<MentsManager>(grid_env, env_size*4);
    manager->mcts_mode = false;
    manager->temp = temp;
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
    run_ments_integration_test(1,1,10000,0.0,2);
}

TEST(Ments_IntegrationTest, easy_grid_world_multithreaded) {
    run_ments_integration_test(2,4,10000,0.0,1,0.5);
}

TEST(Ments_IntegrationTest, easy_grid_world_stochastic) {
    run_ments_integration_test(1,1,10000,0.1,2);
}

TEST(Ments_IntegrationTest, easy_grid_world_stochastic_multithreaded) {
    run_ments_integration_test(2,4,10000,0.1,1,0.5);
}



/**
 * Also run full whack on a simple game to check that the opponent logic all works
 */
void run_ments_game_integration_test(int env_size, int num_trials, int print_tree_depth=0, int decision_timestep=0) {
    shared_ptr<ThtsEnv> game_env = make_shared<TestThtsGameEnv>(env_size);
    shared_ptr<MentsManager> manager = make_shared<MentsManager>(game_env, env_size*4);
    manager->mcts_mode = false;
    manager->is_two_player_game = true;
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
    run_ments_game_integration_test(3, 10000, 4, 0);
}

TEST(Ments_IntegrationTest, two_player_game_env_starting_as_opponent) {
    run_ments_game_integration_test(3, 10000, 4, 1);
}