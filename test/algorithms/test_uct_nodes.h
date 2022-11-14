#pragma once

#include "algorithms/uct_chance_node.h"
#include "algorithms/uct_decision_node.h"

namespace thts_test {
    
    void test_uct_prior();
    void test_uct_compute_ucb_term();
    void test_uct_compute_ucb_values();
    void test_uct_select_action_ucb();
    void test_uct_select_action_random();
    void test_uct_select_action();
    void test_uct_recommend_best_empirical();
    void test_uct_recommend_most_visited();
    void test_uct_backup();
    void test_uct_is_leaf();
    
    /**
     * TODO: docstring
     */
    void run_uct_node_tests();
}