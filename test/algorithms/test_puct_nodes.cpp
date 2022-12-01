#include "test/algorithms/test_puct_nodes.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "algorithms/puct_chance_node.h"
#include "algorithms/puct_decision_node.h"
#include "algorithms/puct_manager.h"

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
    shared_ptr<PuctManager> manager = make_shared<PuctManager>(grid_env, env_size*4);
    manager->mcts_mode = false;
    shared_ptr<PuctDNode> root_node = make_shared<PuctDNode>(manager, grid_env->get_initial_state_itfc(), 0, 0);
    ThtsPool uct_pool(manager, root_node, num_threads);
    uct_pool.run_trials(num_trials);

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